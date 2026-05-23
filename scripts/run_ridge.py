"""Train and evaluate the ridge spike-count baseline across event budgets.

For each (seed, event_budget):
  1. Filter the processed dataset down to the earliest fraction f of
     spike events per bin.
  2. Rebuild the dense spike_counts matrix from those retained events.
  3. Sweep ridge alphas on the validation set, pick the best.
  4. Refit on train with that alpha and score the held-out test set.

Writes two outputs:
  * results/ridge/results.csv             (legacy flat dump)
  * results/ridge/ridge_results.json      (canonical tracking format
                                           the plotting code consumes)

Owned by data-ridge-baseline.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.experiment_runner import append_result, save_json_results
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import restrict_to_event_budget
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_ridge")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)
SEEDS_DEFAULT = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ridge baseline across event budgets.")
    p.add_argument(
        "--processed-path",
        type=Path,
        default=Path("data/processed/processed_mc_rtt.npz"),
    )
    p.add_argument(
        "--results-csv", type=Path, default=Path("results/ridge/results.csv")
    )
    p.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/ridge/ridge_results.json"),
    )
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument(
        "--event-budgets", type=float, nargs="+", default=list(EVENT_BUDGETS_DEFAULT)
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    data = load_processed(args.processed_path)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    y = np.asarray(data["velocity"], dtype=np.float32)
    num_bins, num_neurons = data["spike_counts"].shape
    logger.info(
        "loaded processed data: %d bins, %d neurons (train=%d val=%d test=%d)",
        num_bins,
        num_neurons,
        train_idx.size,
        val_idx.size,
        test_idx.size,
    )

    n_events_total = int(sum(t.size for t in data["event_times"]))
    rows: list[dict] = []

    for seed in args.seeds:
        set_global_seed(seed)
        for f in args.event_budgets:
            logger.info("=" * 72)
            logger.info("ridge: seed=%d event_budget=%.2f", seed, f)
            logger.info("=" * 72)

            sub = restrict_to_event_budget(data, fraction=f)
            X = sub["spike_counts"].astype(np.float32)
            logger.info("X (spike_counts at budget %.2f): shape=%s", f, X.shape)

            decoder = RidgeDecoder(alphas=args.alphas).fit(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            )
            y_pred = decoder.predict(X[test_idx])
            r2 = velocity_r2(y[test_idx], y_pred)
            n_events_used = int(sum(t.size for t in sub["event_times"]))

            logger.info(
                "ridge result: f=%.2f seed=%d  test r2_joint=%+.4f (vx=%+.4f vy=%+.4f)  best_alpha=%g  events=%d/%d",
                f,
                seed,
                r2["r2_joint"],
                r2["r2_vx"],
                r2["r2_vy"],
                decoder.best_alpha,
                n_events_used,
                n_events_total,
            )

            row = {
                "model": "ridge",
                "event_budget": float(f),
                "seed": int(seed),
                "r2_vx": r2["r2_vx"],
                "r2_vy": r2["r2_vy"],
                "r2_joint": r2["r2_joint"],
                "best_alpha": decoder.best_alpha,
                "alpha_sweep": decoder.alpha_sweep,
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": "",
            }
            rows.append(row)
            append_result(args.results_csv, row)

            # Save predictions for the reference seed so the qualitative
            # trajectory figure in scripts/generate_final_figures.py has
            # something to overlay. Other seeds skip this to keep file
            # count bounded.
            if seed == args.seeds[0]:
                pred_path = (
                    Path(args.results_csv).parent
                    / f"predictions_f{f:.2f}_seed{seed}.npz"
                )
                np.savez(
                    pred_path,
                    y_true=y[test_idx],
                    y_pred=y_pred.astype(np.float32),
                    test_idx=test_idx,
                    bin_size_ms=np.array(int(data["bin_size_ms"])),
                    event_budget=np.array(float(f)),
                    model=np.array("ridge"),
                )
                logger.info("saved predictions to %s", pred_path)

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": int(data["bin_size_ms"]),
        "num_neurons": int(data["num_neurons"]),
        "event_budgets": list(args.event_budgets),
        "seeds": list(args.seeds),
        "alphas_swept": list(args.alphas),
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
    }
    save_json_results(args.results_json, model="ridge", config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
