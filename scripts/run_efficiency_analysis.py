"""Run the secondary computational-efficiency analysis for the ridge baseline.

Reads the processed dataset, counts events per bin at every requested
event budget, computes dense and event-driven MAC totals, and writes
`results/ridge/computational_efficiency.json`. This is independent of
ridge training — it only needs the processed dataset to exist.

Owned by data-ridge-baseline.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_processed
from src.evaluation.efficiency_tracker import (
    compute_efficiency_summary,
    save_efficiency_json,
)

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_efficiency_analysis")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ridge baseline efficiency metrics.")
    p.add_argument(
        "--processed-path",
        type=Path,
        default=Path("data/processed/processed_mc_rtt.npz"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/ridge/computational_efficiency.json"),
    )
    p.add_argument(
        "--event-budgets",
        type=float,
        nargs="+",
        default=list(EVENT_BUDGETS_DEFAULT),
        help="event budget fractions to profile",
    )
    p.add_argument(
        "--num-outputs",
        type=int,
        default=2,
        help="number of velocity dimensions decoded (default 2 for vx, vy)",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    data = load_processed(args.processed_path)
    summary = compute_efficiency_summary(
        data,
        fractions=tuple(args.event_budgets),
        num_outputs=args.num_outputs,
        model="ridge",
    )
    save_efficiency_json(summary, args.output)

    # One-line headline per budget for quick paper reference
    logger.info("=" * 72)
    logger.info("efficiency summary (model=ridge, N=%d, T=%d):", summary["num_neurons"], summary["num_bins"])
    for row in summary["budgets"]:
        logger.info(
            "  f=%.2f -> events=%d (%.2f/bin), MACs avoided=%d (%.1f%% of dense)",
            row["event_budget"],
            row["events_total"],
            row["events_per_bin_mean"],
            row["macs_avoided"],
            100.0 * row["macs_avoided_fraction"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
