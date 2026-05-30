"""Trained-SNN deep grid: find the best (k_history x tau x hidden) combo, then
evaluate it across every event budget.

Single-knob sensitivity (results/cluster/snn_sensitivity) showed k_history
and tau_ms are the load-bearing levers (k=16 -> 0.61, tau=40 -> 0.59 at
f=1.0) but never crossed them. This runner sweeps the cross product at
f=1.0 (stage 1, picks the best by val R2), then re-evaluates that single
best configuration across all budgets with more seeds (stage 2).

Writes results/cluster/trained_snn_deep/{grid.csv, best.json, results.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import apply_event_budget
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_trained_snn_grid")


def fit_one(data, event_budget, seed, hidden_dim, tau_ms, threshold, k_history,
            num_sub_bins, epochs, patience, n_boot):
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])
    et, en = apply_event_budget(data["event_times"], data["event_neurons"], event_budget)
    set_global_seed(seed)
    snn = TrainedLatencySNN(
        num_neurons=num_neurons, hidden_dim=hidden_dim, tau_ms=tau_ms,
        threshold=threshold, bin_size_ms=bin_size_ms, k_history=k_history,
        num_sub_bins=num_sub_bins, epochs=epochs, patience=patience, seed=seed,
    ).fit(et, en, y, train_idx, val_idx)
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    y_pred = snn.predict(et, en, test_idx, split_starts=split_starts)
    r2 = velocity_r2(y[test_idx], y_pred)
    r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=n_boot, seed=seed)
    return r2, r2_boot, float(snn.best_val_r2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/cluster/trained_snn_deep"))
    p.add_argument("--k-histories", type=int, nargs="+", default=[16, 20])
    p.add_argument("--taus", type=float, nargs="+", default=[20.0, 40.0])
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 512])
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--num-sub-bins-list", type=int, nargs="+", default=[10],
                   help="sub-bin resolutions to grid; fewer sub-bins shorten the BPTT "
                        "sequence so deeper k_history stays trainable")
    p.add_argument("--stage1-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--stage2-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--stage2-budgets", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_processed(args.processed_path)

    # ---- Stage 1: cross-product grid at f=1.0, pick best by mean val R2 ----
    grid_rows: list[dict] = []
    combo_val: dict[tuple, list[float]] = {}
    logger.info("=" * 72)
    logger.info("STAGE 1: grid k=%s x tau=%s x hidden=%s at f=1.0, seeds=%s",
                args.k_histories, args.taus, args.hidden_dims, args.stage1_seeds)
    logger.info("=" * 72)
    for k_hist, tau, hid, nsb in product(args.k_histories, args.taus, args.hidden_dims,
                                         args.num_sub_bins_list):
        for seed in args.stage1_seeds:
            r2, r2_boot, val_r2 = fit_one(
                data, 1.0, seed, hid, tau, args.threshold, k_hist,
                nsb, args.epochs, args.patience, args.n_boot)
            grid_rows.append({
                "k_history": k_hist, "tau_ms": tau, "hidden_dim": hid, "num_sub_bins": nsb,
                "seed": seed, "r2_joint": r2["r2_joint"], "val_r2": val_r2,
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"], "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
            })
            combo_val.setdefault((k_hist, tau, hid, nsb), []).append(val_r2)
            logger.info("grid k=%d tau=%g hidden=%d nsb=%d seed=%d  test=%+.4f val=%+.4f",
                        k_hist, tau, hid, nsb, seed, r2["r2_joint"], val_r2)
            with (args.out_dir / "grid.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(grid_rows[0].keys())); w.writeheader(); w.writerows(grid_rows)

    best_combo = max(combo_val, key=lambda c: float(np.mean(combo_val[c])))
    best = {"k_history": best_combo[0], "tau_ms": best_combo[1], "hidden_dim": best_combo[2],
            "num_sub_bins": best_combo[3], "mean_val_r2": float(np.mean(combo_val[best_combo]))}
    (args.out_dir / "best.json").write_text(json.dumps(best, indent=2))
    logger.info("STAGE 1 best: k=%d tau=%g hidden=%d nsb=%d (mean val R2=%.4f)",
                best["k_history"], best["tau_ms"], best["hidden_dim"], best["num_sub_bins"],
                best["mean_val_r2"])

    # ---- Stage 2: best combo across all budgets, more seeds ----
    logger.info("=" * 72)
    logger.info("STAGE 2: best combo across budgets=%s seeds=%s", args.stage2_budgets, args.stage2_seeds)
    logger.info("=" * 72)
    final_rows: list[dict] = []
    for f in args.stage2_budgets:
        for seed in args.stage2_seeds:
            r2, r2_boot, val_r2 = fit_one(
                data, f, seed, best["hidden_dim"], best["tau_ms"], args.threshold,
                best["k_history"], best["num_sub_bins"], args.epochs, args.patience, args.n_boot)
            final_rows.append({
                "model": "trained_snn_deep", "event_budget": float(f), "seed": int(seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"], "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "best_val_r2": val_r2, "k_history": best["k_history"], "tau_ms": best["tau_ms"],
                "hidden_dim": best["hidden_dim"], "num_sub_bins": best["num_sub_bins"],
                "n_boot": int(args.n_boot),
            })
            logger.info("final f=%.2f seed=%d  test=%+.4f [%.4f, %.4f]",
                        f, seed, r2["r2_joint"], r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"])
            with (args.out_dir / "results.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(final_rows[0].keys())); w.writeheader(); w.writerows(final_rows)

    logger.info("wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
