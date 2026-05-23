"""Final-figure helpers.

Two paper-grade figures live here:

  * `plot_accuracy_efficiency_frontier` — the headline curve: velocity R²
    vs. event budget f, one line per model, overlay-friendly so the SNN
    and order-shuffle JSONs slot in alongside ridge without code changes.

  * `plot_qualitative_trajectories` — a multi-panel snapshot of true vs.
    predicted 2D cursor velocity at a specific event budget. Reads
    per-model predictions saved by each training script as `.npz` files.

Each model's tracking JSON follows the canonical format written by
`src.evaluation.experiment_runner.save_json_results`. Each model's
predictions follow the schema documented in `plot_qualitative_trajectories`.
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

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "ridge": "Ridge (spike counts)",
    "snn": "Sparse latency SNN",
    "snn_shuffle": "Shuffled-order SNN (control)",
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
    dpi: int = 200,
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
        label = MODEL_DISPLAY_NAMES.get(model, model)
        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            label=label,
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
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (overlaid %d model curves, dpi=%d)", out_path, len(drawn), dpi)


def plot_qualitative_trajectories(
    predictions_paths: dict[str, Path],
    out_path: Path,
    snapshot_seconds: float = 1.5,
    snapshot_start_seconds: float = 0.0,
    event_budget: float | None = None,
    dpi: int = 300,
) -> None:
    """Multi-panel true-vs-predicted 2D velocity snapshot, one row per model.

    Each `.npz` in `predictions_paths` is expected to contain at least:
      * `y_true` — `[num_test_bins, 2]` ground-truth velocity in (vx, vy)
      * `y_pred` — `[num_test_bins, 2]` predicted velocity
      * `bin_size_ms` — scalar, in milliseconds

    Renders two columns (vx, vy) × one row per model so trajectories line
    up vertically for visual comparison. Time axis is the snapshot window
    in seconds; longer recordings are clipped.
    """
    if not predictions_paths:
        raise ValueError("predictions_paths is empty")

    models = list(predictions_paths.keys())
    n_models = len(models)
    fig, axes = plt.subplots(
        n_models,
        2,
        figsize=(11.0, 2.4 * n_models + 0.6),
        sharex=True,
    )
    if n_models == 1:
        axes = np.array(axes).reshape(1, 2)

    for i, model in enumerate(models):
        path = Path(predictions_paths[model])
        if not path.is_file():
            logger.warning("predictions file missing for %s: %s", model, path)
            continue
        z = np.load(path, allow_pickle=True)
        y_true = np.asarray(z["y_true"])
        y_pred = np.asarray(z["y_pred"])
        bin_size_ms = float(z["bin_size_ms"]) if "bin_size_ms" in z.files else 50.0
        if y_true.shape != y_pred.shape or y_true.shape[1] != 2:
            raise ValueError(
                f"{model}: bad prediction shapes y_true={y_true.shape} y_pred={y_pred.shape}"
            )

        bins_per_snapshot = max(2, int(round(snapshot_seconds * 1000.0 / bin_size_ms)))
        start_bin = int(round(snapshot_start_seconds * 1000.0 / bin_size_ms))
        end_bin = min(start_bin + bins_per_snapshot, y_true.shape[0])
        start_bin = max(0, min(start_bin, y_true.shape[0] - 2))
        t_axis = (np.arange(end_bin - start_bin) * bin_size_ms) / 1000.0

        style = MODEL_STYLES.get(model, {"color": "tab:blue"})
        pred_color = style.get("color", "tab:blue")
        label = MODEL_DISPLAY_NAMES.get(model, model)

        for col, dim_name in enumerate(("v_x", "v_y")):
            ax = axes[i, col]
            ax.plot(
                t_axis,
                y_true[start_bin:end_bin, col],
                color="black",
                linewidth=2.0,
                label="true",
            )
            ax.plot(
                t_axis,
                y_pred[start_bin:end_bin, col],
                color=pred_color,
                linewidth=1.6,
                alpha=0.85,
                label="predicted",
            )
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{label}\n{dim_name}")
            else:
                ax.set_ylabel(dim_name)
            if i == 0:
                ax.set_title(f"Cursor velocity {dim_name}")
            if i == 0 and col == 1:
                ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        logger.info(
            "qualitative panel for %s: snapshot bins=[%d, %d), %.2fs window",
            model,
            start_bin,
            end_bin,
            (end_bin - start_bin) * bin_size_ms / 1000.0,
        )

    for col in range(2):
        axes[-1, col].set_xlabel("Time within test snapshot (s)")

    budget_str = f"f = {event_budget:.2f}" if event_budget is not None else ""
    suptitle = "True vs. predicted cursor velocity"
    if budget_str:
        suptitle += f"  ({budget_str})"
    fig.suptitle(suptitle, y=1.00)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s (dpi=%d, %d models)", out_path, dpi, n_models)
