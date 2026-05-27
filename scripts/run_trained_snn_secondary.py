"""Secondary experiments rerun with the trained (BPTT) SNN.

Replicates three analyses that previously ran on the reservoir SNN:

  1. Causal-window sweep: feed only the first window_ms of each bin to
     the trained SNN, refit, score on test.
  2. Permutation test: re-permute within-bin order of events many
     times, refit trained SNN, build null distribution vs real-order
     SNN R^2.
  3. Temporal generalization: fit on early/late/full halves of
     train_idx, evaluate on the canonical test_idx.

Writes:
  * results/trained_snn/causal_window.json
  * results/trained_snn/permutation_test.json
  * results/trained_snn/temporal_generalization.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.controls.order_shuffle import shuffle_within_bin_order
from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.causal_window import truncate_to_window
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_trained_snn_secondary")


def _fit_predict(
    et, en, *, num_neurons, bin_size_ms, hidden_dim, threshold, k_history, epochs, patience,
    velocity, train_idx, val_idx, test_idx, seed,
) -> np.ndarray:
    snn = TrainedLatencySNN(
        num_neurons=num_neurons, hidden_dim=hidden_dim, threshold=threshold,
        bin_size_ms=bin_size_ms, k_history=k_history,
        epochs=epochs, patience=patience, seed=seed,
    ).fit(et, en, velocity, train_idx, val_idx)
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    return snn.predict(et, en, test_idx, split_starts=split_starts)


def causal_window(args, data, y, train_idx, val_idx, test_idx, num_neurons, bin_size_ms):
    out = []
    for w in args.windows_ms:
        logger.info("=" * 72)
        logger.info("trained_snn causal window=%.0f ms", w)
        logger.info("=" * 72)
        et, en = truncate_to_window(data["event_times"], data["event_neurons"], window_ms=w)
        for seed in args.seeds:
            set_global_seed(seed)
            y_pred = _fit_predict(
                et, en,
                num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                hidden_dim=args.hidden_dim, threshold=args.threshold,
                k_history=args.k_history, epochs=args.epochs, patience=args.patience,
                velocity=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=seed,
            )
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
            row = {
                "model": "trained_snn", "window_ms": float(w), "seed": int(seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
            }
            out.append(row)
            logger.info(
                "causal w=%.0f seed=%d  r2=%+.4f [%.4f, %.4f]",
                w, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
            )
    return out


def permutation_test(args, data, y, train_idx, val_idx, test_idx, num_neurons, bin_size_ms):
    set_global_seed(args.seed)
    et = data["event_times"]
    en = data["event_neurons"]
    y_real = _fit_predict(
        et, en,
        num_neurons=num_neurons, bin_size_ms=bin_size_ms,
        hidden_dim=args.hidden_dim, threshold=args.threshold,
        k_history=args.k_history, epochs=args.epochs, patience=args.patience,
        velocity=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=args.seed,
    )
    real_r2 = float(velocity_r2(y[test_idx], y_real)["r2_joint"])
    logger.info("trained_snn permutation: real r2=%+.4f", real_r2)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(10_000, 10_000_000, size=args.n_perm).tolist()
    null_r2s = []
    for i, sd in enumerate(seeds):
        et_s, en_s = shuffle_within_bin_order(et, en, seed=int(sd))
        y_null = _fit_predict(
            et_s, en_s,
            num_neurons=num_neurons, bin_size_ms=bin_size_ms,
            hidden_dim=args.hidden_dim, threshold=args.threshold,
            k_history=args.k_history, epochs=args.epochs, patience=args.patience,
            velocity=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=args.seed,
        )
        null_r2 = float(velocity_r2(y[test_idx], y_null)["r2_joint"])
        null_r2s.append(null_r2)
        if (i + 1) % 5 == 0:
            logger.info("  perm %d/%d null=%+.4f running_mean=%+.4f",
                        i + 1, args.n_perm, null_r2, float(np.mean(null_r2s)))
    nr = np.array(null_r2s, dtype=np.float64)
    p_value = float((nr >= real_r2).sum() + 1) / float(args.n_perm + 1)
    return {
        "real_r2_joint": real_r2,
        "null_r2_joints": null_r2s,
        "null_mean": float(nr.mean()),
        "null_std": float(nr.std(ddof=1)) if args.n_perm > 1 else 0.0,
        "null_lo_ci": float(np.percentile(nr, 2.5)),
        "null_hi_ci": float(np.percentile(nr, 97.5)),
        "p_value_one_sided": p_value,
        "real_minus_null_mean": float(real_r2 - nr.mean()),
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
    }


def temporal_generalization(args, data, y, train_idx, val_idx, test_idx, num_neurons, bin_size_ms):
    out = []
    et = data["event_times"]; en = data["event_neurons"]
    regimes = {
        "early": train_idx[: train_idx.size // 2],
        "late":  train_idx[train_idx.size // 2 :],
        "full":  train_idx,
    }
    for seed in args.seeds:
        for regime, tr in regimes.items():
            set_global_seed(seed)
            logger.info("trained_snn temporal: regime=%s seed=%d", regime, seed)
            y_pred = _fit_predict(
                et, en,
                num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                hidden_dim=args.hidden_dim, threshold=args.threshold,
                k_history=args.k_history, epochs=args.epochs, patience=args.patience,
                velocity=y, train_idx=tr, val_idx=val_idx, test_idx=test_idx, seed=seed,
            )
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
            out.append({
                "model": "trained_snn", "regime": regime,
                "train_size": int(tr.size), "seed": int(seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
            })
            logger.info(
                "  %s seed=%d r2=%+.4f [%.4f, %.4f]",
                regime, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
            )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/trained_snn"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--windows-ms", type=float, nargs="+", default=[10.0, 20.0, 30.0, 40.0, 50.0])
    p.add_argument("--n-perm", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["causal", "permutation", "temporal"])
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])

    if "causal" not in args.skip:
        rows = causal_window(args, data, y, train_idx, val_idx, test_idx, num_neurons, bin_size_ms)
        (args.out_dir / "causal_window.json").write_text(json.dumps({"rows": rows}, indent=2))
        logger.info("wrote causal_window.json (%d rows)", len(rows))

    if "permutation" not in args.skip:
        perm = permutation_test(args, data, y, train_idx, val_idx, test_idx, num_neurons, bin_size_ms)
        (args.out_dir / "permutation_test.json").write_text(json.dumps(perm, indent=2))
        logger.info("wrote permutation_test.json (p=%.4f gap=%+.4f)",
                    perm["p_value_one_sided"], perm["real_minus_null_mean"])

    if "temporal" not in args.skip:
        rows = temporal_generalization(args, data, y, train_idx, val_idx, test_idx, num_neurons, bin_size_ms)
        (args.out_dir / "temporal_generalization.json").write_text(json.dumps({"rows": rows}, indent=2))
        logger.info("wrote temporal_generalization.json (%d rows)", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
