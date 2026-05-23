"""Final-figure helpers.

The headline figure is the accuracy-efficiency frontier: velocity R² on the
y-axis against event budget f on the x-axis, with one line per model.
`plot_accuracy_efficiency_frontier` takes a list of per-model JSON tracking
files (the canonical format written by `save_json_results`) and overlays
each model's curve on a single axis, so the SNN and order-shuffle control
JSON files can be passed in alongside the ridge JSON once the partner
finishes training.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


MODEL_STYLES: dict[str, dict] = {
    "ridge": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "snn": {"color": "#d62728", "marker": "s", "linestyle": "-"},
    "snn_shuffle": {"color": "#7f7f7f", "marker": "x", "linestyle": "--"},
}


def _load_results_json(path: Path) -> tuple[str, list[dict]]:
    """Return (model_name, list_of_result_rows) from a tracking JSON file."""
    with Path(path).open() as f:
        blob = json.load(f)
    model = blob.get("model", Path(path).stem)
    return model, list(blob.get("results", []))


def _aggregate_by_budget(
    rows: list[dict], metric_key: str = "r2_joint"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rows by event_budget; return (budgets, mean, std) sorted descending in f."""
    buckets: dict[float, list[float]] = {}
    for row in rows:
        f = float(row["event_budget"])
        v = row.get(metric_key)
        if v is None:
            continue
        buckets.setdefault(f, []).append(float(v))
    budgets = np.array(sorted(buckets.keys(), reverse=True), dtype=np.float64)
    means = np.array([np.mean(buckets[f]) for f in budgets])
    stds = np.array(
        [np.std(buckets[f], ddof=1) if len(buckets[f]) > 1 else 0.0 for f in budgets]
    )
    return budgets, means, stds


def plot_accuracy_efficiency_frontier(
    results_jsons: Iterable[Path],
    out_path: Path,
    metric_key: str = "r2_joint",
    title: str = "Decoding accuracy vs. sparse event budget",
    y_lim: tuple[float, float] = (-0.05, 1.0),
) -> None:
    """Overlay one R²-vs-event-budget curve per model on a single axis.

    Designed so SNN and order-shuffle curves can be added simply by passing
    their JSON paths alongside the ridge JSON — no code change needed.
    Visual margins are generous on purpose so additional curves don't
    crowd the existing one.
    """
    results_jsons = [Path(p) for p in results_jsons]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    drawn: list[str] = []
    for path in results_jsons:
        if not path.is_file():
            logger.warning("skipping missing results json: %s", path)
            continue
        model, rows = _load_results_json(path)
        if not rows:
            logger.warning("no rows in %s", path)
            continue
        budgets, means, stds = _aggregate_by_budget(rows, metric_key)
        style = MODEL_STYLES.get(model, {"marker": "o", "linestyle": "-"})
        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            label=model,
            capsize=3,
            linewidth=2,
            markersize=7,
            **style,
        )
        drawn.append(model)
        logger.info(
            "plotted %s: budgets=%s means=%s",
            model,
            budgets.tolist(),
            [round(m, 4) for m in means.tolist()],
        )

    ax.set_xlabel("Event budget  f  (fraction of earliest spike events retained)")
    ax.set_ylabel(f"Velocity R²  ({metric_key})")
    ax.set_title(title)
    # Inverted x-axis so "fewer events" goes right — the neuromorphic direction.
    ax.set_xlim(1.05, -0.02)
    ax.set_ylim(*y_lim)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.grid(True, alpha=0.3)
    if drawn:
        ax.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (overlaid %d model curves)", out_path, len(drawn))


# Back-compat shim for scripts that still call the original name.
def plot_r2_vs_event_budget(results_csv: Path, out_path: Path) -> None:
    """Deprecated: prefer plot_accuracy_efficiency_frontier with JSON inputs."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    metric = "r2_joint" if "r2_joint" in df.columns else "r2_mean"
    for model, sub in df.groupby("model"):
        agg = sub.groupby("event_budget")[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["event_budget"], agg["mean"], yerr=agg["std"], label=model, marker="o"
        )
    ax.set_xlabel("Event budget f")
    ax.set_ylabel(f"Velocity R² ({metric})")
    ax.set_title("Decoding accuracy vs. sparse event budget")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
