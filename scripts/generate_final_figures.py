"""Generate the publication-ready final figures from every available model.

Reads (any missing input is skipped with a warning):
  * results/ridge/ridge_results.json              count-ridge curve
  * results/ridge/ridge_lag4_results.json         lag-4 ridge ceiling curve
  * results/latency/latency_results.json          pure-latency ridge curve
  * results/snn/snn_results.json                  reservoir SNN curve
  * results/trained_snn/trained_snn_results.json  trained SNN curve
  * results/controls/shuffle_results.json         order-shuffle control
  * per-model prediction npz files at the qualitative event budget

Writes (all at dpi=300):
  * results/figures/accuracy_efficiency_frontier.png  every available curve overlaid
  * results/figures/velocity_trajectories_f{budget}.png  per-model true-vs-predicted velocity
  * results/figures/cursor_trajectories_f{budget}.png   per-model integrated 2D cursor paths
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.plots import (
    plot_accuracy_efficiency_frontier,
    plot_qualitative_trajectories,
    plot_trajectory_reconstruction,
)

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("generate_final_figures")

TRACKING_JSONS: dict[str, Path] = {
    "ridge": Path("results/ridge/ridge_results.json"),
    "ridge_lag4": Path("results/ridge/ridge_lag4_results.json"),
    "latency": Path("results/latency/latency_results.json"),
    "snn": Path("results/snn/snn_results.json"),
    "trained_snn": Path("results/trained_snn/trained_snn_results.json"),
    "snn_shuffle": Path("results/controls/shuffle_results.json"),
}

QUALITATIVE_BUDGET = 0.25
QUALITATIVE_SEED = 0
_npz = lambda d: Path(f"{d}/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz")
PREDICTION_NPZS: dict[str, Path] = {
    "ridge": _npz("results/ridge"),
    "latency": _npz("results/latency"),
    "snn": _npz("results/snn"),
    "trained_snn": _npz("results/trained_snn"),
    "snn_shuffle": _npz("results/controls"),
}

FIGURES_DIR = Path("results/figures")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication-ready final figures.")
    p.add_argument("--qualitative-budget", type=float, default=QUALITATIVE_BUDGET)
    p.add_argument("--snapshot-seconds", type=float, default=1.5)
    p.add_argument("--snapshot-start-seconds", type=float, default=0.0)
    p.add_argument("--trajectory-seconds", type=float, default=6.0,
                   help="length of the 2D trajectory snapshot for the cursor-path figure")
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

    tracking_present = _available(TRACKING_JSONS, "tracking JSON")
    if tracking_present:
        plot_accuracy_efficiency_frontier(
            results_jsons=list(tracking_present.values()),
            out_path=args.figures_dir / "accuracy_efficiency_frontier.png",
            title="Sparse Event-Based BCI Decoding — Accuracy vs. Event Budget",
            dpi=args.dpi,
        )

    preds_present = _available(PREDICTION_NPZS, "prediction npz")
    f_str = f"{args.qualitative_budget:.2f}".replace(".", "p")
    if preds_present:
        plot_qualitative_trajectories(
            predictions_paths=preds_present,
            out_path=args.figures_dir / f"velocity_trajectories_f{f_str}.png",
            snapshot_seconds=args.snapshot_seconds,
            snapshot_start_seconds=args.snapshot_start_seconds,
            event_budget=args.qualitative_budget,
            dpi=args.dpi,
        )
        plot_trajectory_reconstruction(
            predictions_paths=preds_present,
            out_path=args.figures_dir / f"cursor_trajectories_f{f_str}.png",
            snapshot_seconds=args.trajectory_seconds,
            snapshot_start_seconds=args.snapshot_start_seconds,
            event_budget=args.qualitative_budget,
            dpi=args.dpi,
        )

    logger.info(
        "final figures complete; %d tracking JSON(s), %d prediction npz(s) used",
        len(tracking_present), len(preds_present),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
