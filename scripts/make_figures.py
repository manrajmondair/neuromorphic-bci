"""Build the headline accuracy-efficiency frontier figure.

Reads per-model JSON tracking files (the canonical format written by
`src.evaluation.experiment_runner.save_json_results`) and overlays each
model's R²-vs-event-budget curve on a single axis. Designed so the SNN
and order-shuffle JSONs slot in alongside ridge with no code change.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.evaluation.plots import plot_accuracy_efficiency_frontier

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("make_figures")

DEFAULT_JSONS = (
    Path("results/ridge/ridge_results.json"),
    Path("results/snn/snn_results.json"),
    Path("results/snn/snn_shuffle_results.json"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render the accuracy-efficiency frontier figure.")
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_JSONS),
        help="per-model JSON tracking files to overlay",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/r2_vs_event_budget.png"),
    )
    p.add_argument(
        "--metric",
        default="r2_joint",
        help="which key in the result rows to plot (default: r2_joint, the proposal formula)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    available = [p for p in args.inputs if p.is_file()]
    if not available:
        logger.error(
            "no result JSON files found; expected any of: %s",
            ", ".join(str(p) for p in args.inputs),
        )
        return 1
    missing = [p for p in args.inputs if not p.is_file()]
    if missing:
        logger.info("skipping missing inputs (will be overlaid once they exist): %s", missing)

    plot_accuracy_efficiency_frontier(
        results_jsons=available,
        out_path=args.output,
        metric_key=args.metric,
    )
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
