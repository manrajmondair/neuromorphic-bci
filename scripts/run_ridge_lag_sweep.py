"""Sweep ridge lag_bins ∈ {0, 1, 2, 4, 8, 16} across event budgets.

Strengthens the "ridge_lag4 is a strong baseline" claim by showing the
lag-vs-accuracy curve plateaus around 4 — i.e., adding more history
beyond 200 ms doesn't help. If the curve keeps rising, the paper's
single-lag claim is too weak and we'd need to report the better one.

Writes results/cluster/ridge_lag_sweep/{summary.json, results.csv}.
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
from src.features.event_budget import restrict_to_event_budget
from src.features.spike_counts import stack_lag_features
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_ridge_lag_sweep")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/cluster/ridge_lag_sweep"))
    p.add_argument("--lag-bins", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.50, 0.25, 0.10])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))

    budget_cache = {
        float(f): restrict_to_event_budget(data, fraction=f)["spike_counts"].astype(np.float32)
        for f in args.event_budgets
    }

    rows: list[dict] = []
    for lag in args.lag_bins:
        for f in args.event_budgets:
            spike_counts = budget_cache[float(f)]
            X = stack_lag_features(spike_counts, num_lags=lag, split_starts=split_starts)
            for seed in args.seeds:
                set_global_seed(seed)
                decoder = RidgeDecoder(alphas=DEFAULT_ALPHAS).fit(
                    X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                )
                y_pred = decoder.predict(X[test_idx])
                r2 = velocity_r2(y[test_idx], y_pred)
                r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
                row = {
                    "model": "ridge_lag_sweep",
                    "lag_bins": int(lag),
                    "event_budget": float(f),
                    "seed": int(seed),
                    "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                    "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                    "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                    "best_alpha": float(decoder.best_alpha),
                    "n_features": int(X.shape[1]),
                    "n_boot": int(args.n_boot),
                }
                rows.append(row)
                logger.info(
                    "lag=%2d  f=%.2f  seed=%d  r2_joint=%+.4f [%.4f, %.4f]  alpha=%g  features=%d",
                    lag, f, seed, r2["r2_joint"],
                    r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
                    decoder.best_alpha, X.shape[1],
                )

    with (args.out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (args.out_dir / "summary.json").write_text(json.dumps({
        "model": "ridge_lag_sweep",
        "dataset": "NLB_MC_RTT",
        "metric": "velocity_r2",
        "config": {
            "lag_bins": list(args.lag_bins),
            "event_budgets": list(args.event_budgets),
            "seeds": list(args.seeds),
            "n_boot": int(args.n_boot),
        },
        "results": rows,
    }, indent=2))
    logger.info("wrote %s and %s", args.out_dir / "results.csv", args.out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
