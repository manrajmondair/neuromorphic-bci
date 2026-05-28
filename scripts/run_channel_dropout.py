"""Test-time channel dropout — simulate post-deployment electrode failure.

Train ridge_lag4 and trained_snn on clean data ONCE. Then evaluate on a
test split where a random fraction `p` of the 98 channels has been
zeroed out, repeated across K independent channel masks per fraction so
we get mean ± std across masks. Sweep p in {0.0, 0.1, 0.2, 0.3, 0.5}.

This is the implantable-BCI relevance test: real arrays lose channels
over months. A decoder that degrades gracefully under channel dropout
is more deployable than one that breaks. The expectation is that the
trained SNN — which has learned distributed temporal features — should
hold up better than ridge_lag4 which is heavily reliant on each
neuron's count contribution to the linear projection.

Writes `results/cluster/channel_dropout/results.json`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2
from src.features.spike_counts import stack_lag_features
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_channel_dropout")


def _drop_channels_in_counts(counts: np.ndarray, drop_mask: np.ndarray) -> np.ndarray:
    """Zero out the columns of [num_bins, num_neurons] indicated by `drop_mask`."""
    out = counts.copy()
    out[:, drop_mask] = 0
    return out


def _drop_channels_in_events(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    drop_mask: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Remove any event whose neuron id is in the drop_mask."""
    dropped = set(int(x) for x in np.flatnonzero(drop_mask))
    out_t: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    for times, neurons in zip(event_times, event_neurons):
        if times.size == 0 or not dropped:
            out_t.append(times)
            out_n.append(neurons)
            continue
        keep = np.array([int(n) not in dropped for n in neurons], dtype=bool)
        out_t.append(times[keep])
        out_n.append(neurons[keep])
    return out_t, out_n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path,
                   default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("results/cluster/channel_dropout/results.json"))
    p.add_argument("--dropout-fractions", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--n-masks", type=int, default=10,
                   help="independent random masks per dropout fraction")
    p.add_argument("--event-budget", type=float, default=1.00)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--skip-snn", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])
    spike_counts = np.asarray(data["spike_counts"], dtype=np.float32)

    set_global_seed(args.train_seed)

    # Train ridge_lag4 once on clean data; reuse trained weights at every dropout.
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    X_clean = stack_lag_features(spike_counts, num_lags=4, split_starts=split_starts)
    ridge = RidgeDecoder(alphas=DEFAULT_ALPHAS).fit(
        X_clean[train_idx], y[train_idx], X_clean[val_idx], y[val_idx],
    )
    logger.info("trained ridge_lag4: alpha=%g  val r2_joint=%+.4f",
                ridge.best_alpha,
                velocity_r2(y[val_idx], ridge.predict(X_clean[val_idx]))["r2_joint"])

    snn = None
    if not args.skip_snn:
        snn = TrainedLatencySNN(
            num_neurons=num_neurons, hidden_dim=args.hidden_dim,
            threshold=args.threshold, bin_size_ms=bin_size_ms,
            k_history=args.k_history, epochs=args.epochs,
            patience=args.patience, seed=args.train_seed,
        ).fit(data["event_times"], data["event_neurons"], y, train_idx, val_idx)
        logger.info("trained trained_snn: best_val_r2=%+.4f", snn.best_val_r2)

    rows: list[dict] = []
    for frac in args.dropout_fractions:
        n_drop = int(round(frac * num_neurons))
        logger.info("=" * 72)
        logger.info("dropout_fraction=%.2f -> drop %d/%d channels",
                    frac, n_drop, num_neurons)
        logger.info("=" * 72)

        for mask_seed in range(args.n_masks):
            rng = np.random.default_rng(args.train_seed * 1000 + mask_seed + 1)
            drop_mask = np.zeros(num_neurons, dtype=bool)
            if n_drop > 0:
                drop_idx = rng.choice(num_neurons, size=n_drop, replace=False)
                drop_mask[drop_idx] = True

            # Ridge: zero out the dropped channels in spike_counts -> re-stack lag features.
            counts_drop = _drop_channels_in_counts(spike_counts, drop_mask)
            X_drop = stack_lag_features(counts_drop, num_lags=4, split_starts=split_starts)
            y_pred_ridge = ridge.predict(X_drop[test_idx])
            r2_ridge = velocity_r2(y[test_idx], y_pred_ridge)

            row_ridge = {
                "model": "ridge_lag4",
                "dropout_fraction": float(frac),
                "n_dropped": int(n_drop),
                "mask_seed": int(mask_seed),
                "r2_vx": r2_ridge["r2_vx"],
                "r2_vy": r2_ridge["r2_vy"],
                "r2_joint": r2_ridge["r2_joint"],
            }
            rows.append(row_ridge)

            if snn is not None:
                et_drop, en_drop = _drop_channels_in_events(
                    data["event_times"], data["event_neurons"], drop_mask,
                )
                y_pred_snn = snn.predict(et_drop, en_drop, test_idx, split_starts=split_starts)
                r2_snn = velocity_r2(y[test_idx], y_pred_snn)
                rows.append({
                    "model": "trained_snn",
                    "dropout_fraction": float(frac),
                    "n_dropped": int(n_drop),
                    "mask_seed": int(mask_seed),
                    "r2_vx": r2_snn["r2_vx"],
                    "r2_vy": r2_snn["r2_vy"],
                    "r2_joint": r2_snn["r2_joint"],
                })
                logger.info(
                    "p=%.2f mask=%d  ridge=%+.4f  trained_snn=%+.4f",
                    frac, mask_seed, r2_ridge["r2_joint"], r2_snn["r2_joint"],
                )
            else:
                logger.info("p=%.2f mask=%d  ridge=%+.4f", frac, mask_seed, r2_ridge["r2_joint"])

            args.out.write_text(json.dumps({"rows": rows}, indent=2))

    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
