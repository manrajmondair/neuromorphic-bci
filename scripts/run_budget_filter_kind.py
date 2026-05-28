"""Test whether 'earliest' event ordering matters — earliest vs random vs latest.

The default event-budget filter keeps the first `f * N` events per bin in
chronological order. This script repeats the headline trained-SNN and
ridge-lag-4 experiment with three different filter rules:

  * `earliest`  the standard filter, keeps the first f*N events
  * `random`    keep f*N events drawn uniformly without replacement
                (re-sorted into ascending time)
  * `latest`    keep the LAST f*N events

If 'random' achieves the same R² as 'earliest', then the temporal
structure exploited by the decoders is not specifically about early
events — within-bin order alone explains the signal. If 'earliest'
beats 'random', then movement-initiation timing carries information
above and beyond order. This is the directly testable form of an
assumption that's implicit throughout the paper.

Writes `results/cluster/budget_filter_kind/results.json`.
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
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import apply_event_budget
from src.features.spike_counts import counts_from_events, stack_lag_features
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_budget_filter_kind")


def _filter_events(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    fraction: float,
    kind: str,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Subselect events per bin under one of three rules.

    `earliest` matches the default `apply_event_budget` (uses time order).
    `latest` keeps the last `k = max(1, floor(fraction * N))` per bin.
    `random` draws `k` indices uniformly without replacement and then
    re-sorts by time so the strict-monotonic invariant survives.
    """
    if kind == "earliest":
        return apply_event_budget(event_times, event_neurons, fraction)
    if fraction == 1.0:
        return event_times, event_neurons
    rng = np.random.default_rng(seed)
    out_t: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    for times, neurons in zip(event_times, event_neurons):
        n = times.size
        if n == 0:
            out_t.append(times)
            out_n.append(neurons)
            continue
        k = max(1, int(np.floor(fraction * n)))
        if kind == "latest":
            idx = np.arange(n - k, n)
        elif kind == "random":
            idx = rng.choice(n, size=k, replace=False)
            idx.sort()
        else:
            raise ValueError(f"unknown kind: {kind}")
        out_t.append(times[idx])
        out_n.append(neurons[idx])
    return out_t, out_n


def _fit_ridge_lag4(spike_counts, y, train_idx, val_idx, test_idx, n_boot, seed):
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    X = stack_lag_features(spike_counts.astype(np.float32), num_lags=4,
                           split_starts=split_starts)
    decoder = RidgeDecoder(alphas=DEFAULT_ALPHAS).fit(
        X[train_idx], y[train_idx], X[val_idx], y[val_idx],
    )
    y_pred = decoder.predict(X[test_idx])
    r2 = velocity_r2(y[test_idx], y_pred)
    r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=n_boot, seed=seed)
    return r2, r2_boot, decoder.best_alpha


def _fit_trained_snn(et, en, y, train_idx, val_idx, test_idx,
                     num_neurons, bin_size_ms, hidden_dim, k_history,
                     threshold, epochs, patience, seed, n_boot):
    snn = TrainedLatencySNN(
        num_neurons=num_neurons, hidden_dim=hidden_dim,
        threshold=threshold, bin_size_ms=bin_size_ms,
        k_history=k_history, epochs=epochs, patience=patience, seed=seed,
    ).fit(et, en, y, train_idx, val_idx)
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    y_pred = snn.predict(et, en, test_idx, split_starts=split_starts)
    r2 = velocity_r2(y[test_idx], y_pred)
    r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=n_boot, seed=seed)
    return r2, r2_boot, snn.best_val_r2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("results/cluster/budget_filter_kind/results.json"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.50, 0.25, 0.10])
    p.add_argument("--kinds", type=str, nargs="+",
                   default=["earliest", "random", "latest"],
                   choices=["earliest", "random", "latest"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--skip-snn", action="store_true",
                   help="ridge_lag4 only (much faster, useful for quick previews)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])

    rows: list[dict] = []
    for f in args.event_budgets:
        for kind in args.kinds:
            for seed in args.seeds:
                set_global_seed(seed)
                et, en = _filter_events(data["event_times"], data["event_neurons"],
                                        fraction=f, kind=kind, seed=seed)
                # Rebuild spike_counts from the filtered events for the ridge side.
                spike_counts = counts_from_events(en, num_neurons)
                logger.info("=" * 72)
                logger.info("f=%.2f kind=%s seed=%d events=%d/%d",
                            f, kind, seed, int(sum(t.size for t in et)),
                            int(sum(t.size for t in data["event_times"])))
                logger.info("=" * 72)

                r2, r2_boot, alpha = _fit_ridge_lag4(
                    spike_counts, y, train_idx, val_idx, test_idx,
                    n_boot=args.n_boot, seed=seed,
                )
                row = {
                    "model": "ridge_lag4",
                    "event_budget": float(f),
                    "kind": kind,
                    "seed": int(seed),
                    "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                    "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                    "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                    "n_boot": int(args.n_boot),
                    "best_alpha": alpha,
                }
                rows.append(row)
                logger.info("ridge_lag4 r2=%+.4f [%.4f, %.4f]", r2["r2_joint"],
                            r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"])

                if not args.skip_snn:
                    r2_snn, r2_snn_boot, val_r2 = _fit_trained_snn(
                        et, en, y, train_idx, val_idx, test_idx,
                        num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                        hidden_dim=args.hidden_dim, k_history=args.k_history,
                        threshold=args.threshold, epochs=args.epochs,
                        patience=args.patience, seed=seed, n_boot=args.n_boot,
                    )
                    rows.append({
                        "model": "trained_snn",
                        "event_budget": float(f),
                        "kind": kind,
                        "seed": int(seed),
                        "r2_vx": r2_snn["r2_vx"], "r2_vy": r2_snn["r2_vy"],
                        "r2_joint": r2_snn["r2_joint"],
                        "r2_joint_ci_lo": r2_snn_boot["r2_joint_ci_lo"],
                        "r2_joint_ci_hi": r2_snn_boot["r2_joint_ci_hi"],
                        "n_boot": int(args.n_boot),
                        "best_val_r2": float(val_r2),
                    })
                    logger.info(
                        "trained_snn r2=%+.4f [%.4f, %.4f] val=%+.4f",
                        r2_snn["r2_joint"], r2_snn_boot["r2_joint_ci_lo"],
                        r2_snn_boot["r2_joint_ci_hi"], val_r2,
                    )

                # Stream after each (budget, kind, seed) cell.
                args.out.write_text(json.dumps({"rows": rows}, indent=2))

    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
