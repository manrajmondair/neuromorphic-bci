"""Trained-SNN hyperparameter sensitivity: vary one knob at a time around the canonical setting.

Three single-knob sweeps at fixed (hidden_dim=256, k_history=4, f=1.0, 3 seeds):
  * tau_ms ∈ {5, 10, 20, 40} — membrane time constant
  * threshold ∈ {0.15, 0.30, 0.50, 0.80} — LIF firing threshold
  * k_history ∈ {0, 2, 4, 8} — input context length

Together they answer "is the trained-SNN result load-bearing on any
particular hyperparameter setting?". A flat response across knobs is
strong evidence that the trained-SNN claim isn't a cherry-picked
operating point.

Writes results/cluster/snn_sensitivity/{summary.json, results.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import apply_event_budget
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_snn_sensitivity")

# Canonical setting that every individual sweep falls back to.
CANONICAL = {
    "hidden_dim": 256,
    "tau_ms": 10.0,
    "threshold": 0.30,
    "k_history": 4,
}


def _fit_one(data, event_budget, seed, params, n_boot):
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])
    et, en = apply_event_budget(data["event_times"], data["event_neurons"], event_budget)

    set_global_seed(seed)
    snn = TrainedLatencySNN(
        num_neurons=num_neurons,
        hidden_dim=params["hidden_dim"],
        tau_ms=params["tau_ms"],
        threshold=params["threshold"],
        bin_size_ms=bin_size_ms,
        k_history=params["k_history"],
        epochs=80, patience=15, seed=seed,
    ).fit(et, en, y, train_idx, val_idx)
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    y_pred = snn.predict(et, en, test_idx, split_starts=split_starts)
    r2 = velocity_r2(y[test_idx], y_pred)
    r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=n_boot, seed=seed)
    return r2, r2_boot, snn.best_val_r2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/cluster/snn_sensitivity"))
    p.add_argument("--event-budget", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--taus", type=float, nargs="+", default=[5.0, 10.0, 20.0, 40.0])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.30, 0.50, 0.80])
    p.add_argument("--k-histories", type=int, nargs="+", default=[0, 2, 4, 8])
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    rows: list[dict] = []

    def sweep(knob_name: str, values: list, value_key: str):
        for v in values:
            for seed in args.seeds:
                params = dict(CANONICAL)
                params[value_key] = v
                r2, r2_boot, val_r2 = _fit_one(data, args.event_budget, seed, params, args.n_boot)
                row = {
                    "knob": knob_name, "value": float(v) if not isinstance(v, int) else int(v),
                    "seed": int(seed), "event_budget": float(args.event_budget),
                    "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                    "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                    "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                    "best_val_r2": float(val_r2),
                    **{f"param_{k}": params[k] for k in params},
                }
                rows.append(row)
                logger.info("knob=%s val=%s seed=%d r2=%+.4f [%.4f, %.4f]",
                            knob_name, v, seed, r2["r2_joint"],
                            r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"])
                # Stream after every cell so a crash leaves usable partial output.
                with (args.out_dir / "results.csv").open("w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    logger.info("=== tau_ms sweep ===")
    sweep("tau_ms", args.taus, "tau_ms")
    logger.info("=== threshold sweep ===")
    sweep("threshold", args.thresholds, "threshold")
    logger.info("=== k_history sweep ===")
    sweep("k_history", args.k_histories, "k_history")

    (args.out_dir / "summary.json").write_text(json.dumps({
        "model": "trained_snn_sensitivity",
        "dataset": "NLB_MC_RTT",
        "metric": "velocity_r2",
        "config": {
            "canonical": CANONICAL,
            "event_budget": float(args.event_budget),
            "seeds": list(args.seeds),
            "taus": list(args.taus),
            "thresholds": list(args.thresholds),
            "k_histories": list(args.k_histories),
            "n_boot": int(args.n_boot),
        },
        "results": rows,
    }, indent=2))
    logger.info("wrote %s and %s", args.out_dir / "results.csv", args.out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
