"""Train and evaluate the ridge spike-count baseline across event budgets.

For each (seed, event_budget):
  1. Filter the processed dataset down to the earliest fraction f of
     spike events per bin.
  2. Rebuild the dense spike_counts matrix from those retained events.
  3. Optionally stack `--lag-bins` previous bins along the feature axis,
     zero-padding across split boundaries so val/test features never
     read across into train.
  4. Sweep ridge alphas on the validation set, pick the best.
  5. Refit on train with that alpha and score the held-out test set.
  6. Bootstrap 95% CIs on test R² by resampling test bins with replacement.

Writes:
  * results/ridge/results.csv             (legacy flat dump)
  * results/ridge/ridge_results.json      (canonical tracking format the
                                           plotting code consumes)

Owned by data-ridge-baseline.
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
from src.features.event_budget import restrict_to_event_budget
from src.features.spike_counts import stack_lag_features
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_ridge")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)
# Ridge is closed-form on a deterministic split + earliest-events filter,
# so multiple seeds produce identical R². The seed CLI still drives the
# bootstrap resampling RNG so its effect is visible on the CI bounds.
SEEDS_DEFAULT = (0,)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ridge baseline across event budgets.")
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/ridge/results.csv"))
    p.add_argument("--results-json", type=Path, default=Path("results/ridge/ridge_results.json"))
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument("--event-budgets", type=float, nargs="+", default=list(EVENT_BUDGETS_DEFAULT))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument(
        "--lag-bins",
        type=int,
        default=0,
        help="History length: stack current bin with the previous k bins along the feature axis",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="Override the 'model' string in the result rows (e.g. 'ridge_lag4')",
    )
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap reps for CI bounds")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    data = load_processed(args.processed_path)
    if args.results_csv.exists():
        args.results_csv.unlink()

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    y = np.asarray(data["velocity"], dtype=np.float32)
    num_bins, num_neurons = data["spike_counts"].shape
    logger.info(
        "loaded processed data: %d bins, %d neurons (train=%d val=%d test=%d) lag_bins=%d",
        num_bins, num_neurons, train_idx.size, val_idx.size, test_idx.size, args.lag_bins,
    )

    # Split-boundary starts for lag-feature zero-padding.
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))

    model_name = args.model_name or ("ridge" if args.lag_bins == 0 else f"ridge_lag{args.lag_bins}")
    n_events_total = int(sum(t.size for t in data["event_times"]))
    rows: list[dict] = []

    for seed in args.seeds:
        set_global_seed(seed)
        for f in args.event_budgets:
            logger.info("=" * 72)
            logger.info("%s: seed=%d event_budget=%.2f", model_name, seed, f)
            logger.info("=" * 72)

            sub = restrict_to_event_budget(data, fraction=f)
            X = stack_lag_features(
                sub["spike_counts"], num_lags=args.lag_bins, split_starts=split_starts
            )
            logger.info("X (lag=%d, budget %.2f): shape=%s", args.lag_bins, f, X.shape)

            decoder = RidgeDecoder(alphas=args.alphas).fit(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            )
            y_pred = decoder.predict(X[test_idx])
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
            n_events_used = int(sum(t.size for t in sub["event_times"]))

            logger.info(
                "%s result: f=%.2f seed=%d  r2_joint=%+.4f [%.4f, %.4f]  best_alpha=%g  events=%d/%d",
                model_name, f, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
                decoder.best_alpha, n_events_used, n_events_total,
            )

            row = {
                "model": model_name,
                "event_budget": float(f),
                "seed": int(seed),
                "r2_vx": r2["r2_vx"],
                "r2_vy": r2["r2_vy"],
                "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "r2_vx_ci_lo": r2_boot["r2_vx_ci_lo"],
                "r2_vx_ci_hi": r2_boot["r2_vx_ci_hi"],
                "r2_vy_ci_lo": r2_boot["r2_vy_ci_lo"],
                "r2_vy_ci_hi": r2_boot["r2_vy_ci_hi"],
                "n_boot": int(args.n_boot),
                "best_alpha": decoder.best_alpha,
                "alpha_sweep": decoder.alpha_sweep,
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": f"lag_bins={args.lag_bins}",
            }
            rows.append(row)
            append_result(args.results_csv, row)

            if seed == args.seeds[0]:
                pred_path = (
                    Path(args.results_csv).parent / f"predictions_f{f:.2f}_seed{seed}.npz"
                )
                np.savez(
                    pred_path,
                    y_true=y[test_idx],
                    y_pred=y_pred.astype(np.float32),
                    test_idx=test_idx,
                    bin_size_ms=np.array(int(data["bin_size_ms"])),
                    event_budget=np.array(float(f)),
                    model=np.array(model_name),
                )
                logger.info("saved predictions to %s", pred_path)

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": int(data["bin_size_ms"]),
        "num_neurons": int(data["num_neurons"]),
        "event_budgets": list(args.event_budgets),
        "seeds": list(args.seeds),
        "alphas_swept": list(args.alphas),
        "lag_bins": int(args.lag_bins),
        "n_boot": int(args.n_boot),
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
    }
    save_json_results(args.results_json, model=model_name, config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
