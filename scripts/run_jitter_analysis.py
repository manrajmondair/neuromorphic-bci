"""Spike-time jitter analysis — how much timing precision does the SNN need?

For each (sigma_ms in σ-grid) × (event_budget) we add gaussian noise σ to
every spike's within-bin time (then re-sort to preserve the data-interface
invariant), re-fit the reservoir SNN with per-budget threshold tuning,
and score test R^2 with bootstrap CIs.

Writes results/jitter/jitter_results.json with one row per (sigma_ms, budget, seed).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.experiment_runner import append_result, save_json_results
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import apply_event_budget
from src.features.jitter import jitter_event_times
from src.models.snn_decoder import SparseLatencySNN, tune_threshold_on_val
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_jitter_analysis")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/jitter/results.csv"))
    p.add_argument("--results-json", type=Path, default=Path("results/jitter/jitter_results.json"))
    p.add_argument("--sigmas-ms", type=float, nargs="+", default=[0.0, 1.0, 2.5, 5.0, 10.0, 20.0])
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.25])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-restarts", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    if args.results_csv.exists():
        args.results_csv.unlink()
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])

    rows = []
    for seed in args.seeds:
        set_global_seed(seed)
        for f in args.event_budgets:
            et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
            # Tune threshold once on the un-jittered budget (so jitter
            # comparisons don't confound with threshold drift).
            best_thr, _ = tune_threshold_on_val(
                num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                event_times=et, event_neurons=en,
                velocity=y, train_idx=train_idx, val_idx=val_idx,
                hidden_dim=args.hidden_dim, n_restarts=1, seed=seed,
            )
            for sigma_ms in args.sigmas_ms:
                logger.info("=" * 72)
                logger.info("jitter: seed=%d budget=%.2f sigma=%g ms thr=%g",
                            seed, f, sigma_ms, best_thr)
                logger.info("=" * 72)
                et_j, en_j = jitter_event_times(et, en, sigma_ms=sigma_ms,
                                                bin_size_ms=bin_size_ms, seed=seed)
                snn = SparseLatencySNN(
                    num_neurons=num_neurons, hidden_dim=args.hidden_dim,
                    threshold=best_thr, bin_size_ms=bin_size_ms,
                    n_restarts=args.n_restarts, standardize=True, seed=seed,
                ).fit(et_j, en_j, y, train_idx, val_idx)
                y_pred = snn.predict(et_j, en_j, test_idx)
                r2 = velocity_r2(y[test_idx], y_pred)
                r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
                logger.info(
                    "result: sigma=%g ms f=%.2f seed=%d  r2_joint=%+.4f [%.4f, %.4f]",
                    sigma_ms, f, seed, r2["r2_joint"],
                    r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
                )
                row = {
                    "model": "snn_jitter",
                    "sigma_ms": float(sigma_ms),
                    "event_budget": float(f), "seed": int(seed),
                    "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                    "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                    "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                    "n_boot": int(args.n_boot),
                    "tuned_threshold": float(best_thr),
                    "notes": "gaussian jitter on within-bin spike times",
                }
                rows.append(row)
                append_result(args.results_csv, row)

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms, "num_neurons": num_neurons,
        "sigmas_ms": list(args.sigmas_ms),
        "event_budgets": list(args.event_budgets),
        "seeds": list(args.seeds),
        "hidden_dim": int(args.hidden_dim),
        "n_restarts": int(args.n_restarts),
        "n_boot": int(args.n_boot),
    }
    save_json_results(args.results_json, model="snn_jitter", config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
