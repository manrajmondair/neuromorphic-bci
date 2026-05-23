"""Generate the publication-ready final figures from all three models.

Reads:
  * results/ridge/ridge_results.json
  * results/snn/snn_results.json
  * results/controls/shuffle_results.json
  * per-model prediction npz files at the qualitative event budget

Writes (all at dpi=300):
  * results/figures/accuracy_efficiency_frontier.png
  * results/figures/velocity_trajectories_f0.25.png

Any missing input is skipped with a warning so the figure can be regenerated
incrementally as each model finishes. The frontier figure works with one or
more models; the qualitative figure works with whatever prediction files
are present.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.evaluation.plots import (
    plot_accuracy_efficiency_frontier,
    plot_qualitative_trajectories,
)

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("generate_final_figures")

# Canonical per-model result locations. Adjust here if a path moves.
TRACKING_JSONS: dict[str, Path] = {
    "ridge": Path("results/ridge/ridge_results.json"),
    "snn": Path("results/snn/snn_results.json"),
    "snn_shuffle": Path("results/controls/shuffle_results.json"),
}

# Per-model prediction npz files used by the qualitative trajectory panel.
QUALITATIVE_BUDGET = 0.25
QUALITATIVE_SEED = 0
PREDICTION_NPZS: dict[str, Path] = {
    "ridge": Path(f"results/ridge/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"),
    "snn": Path(f"results/snn/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"),
    "snn_shuffle": Path(
        f"results/controls/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"
    ),
}

FIGURES_DIR = Path("results/figures")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication-ready final figures.")
    p.add_argument(
        "--qualitative-budget",
        type=float,
        default=QUALITATIVE_BUDGET,
        help="event budget at which to draw the qualitative trajectory panel",
    )
    p.add_argument(
        "--snapshot-seconds",
        type=float,
        default=1.5,
        help="length of the trajectory snapshot in seconds (1-2 s recommended)",
    )
    p.add_argument(
        "--snapshot-start-seconds",
        type=float,
        default=0.0,
        help="offset into the test split where the snapshot starts",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _available(paths: dict[str, Path], kind: str) -> dict[str, Path]:
    present = {k: v for k, v in paths.items() if v.is_file()}
    missing = [(k, v) for k, v in paths.items() if not v.is_file()]
    for k, v in missing:
        logger.warning("%s: missing %s -> %s", kind, k, v)
    return present


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy-efficiency frontier across whichever models are present
    tracking_present = _available(TRACKING_JSONS, "tracking JSON")
    if tracking_present:
        frontier_path = args.figures_dir / "accuracy_efficiency_frontier.png"
        plot_accuracy_efficiency_frontier(
            results_jsons=list(tracking_present.values()),
            out_path=frontier_path,
            title="Sparse Event-Based BCI Decoding — Accuracy vs. Event Budget",
            dpi=args.dpi,
        )
    else:
        logger.error("no tracking JSON files found; skipping frontier plot")

    # 2. Qualitative true-vs-predicted velocity trajectory panel
    preds_present = _available(PREDICTION_NPZS, "prediction npz")
    if preds_present:
        f_str = f"{args.qualitative_budget:.2f}".replace(".", "p")
        traj_path = args.figures_dir / f"velocity_trajectories_f{f_str}.png"
        plot_qualitative_trajectories(
            predictions_paths=preds_present,
            out_path=traj_path,
            snapshot_seconds=args.snapshot_seconds,
            snapshot_start_seconds=args.snapshot_start_seconds,
            event_budget=args.qualitative_budget,
            dpi=args.dpi,
        )
    else:
        logger.error("no prediction npz files found; skipping trajectory plot")

    logger.info(
        "final figures complete; %d tracking JSON(s), %d prediction npz(s) used",
        len(tracking_present),
        len(preds_present),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
