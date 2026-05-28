"""Aggregate all cluster-round result JSONs into one summary + figure refresh.

Consumes whichever of these exist under results/cluster/:
  * block_cv/block_cv.json             (10-seed × 4-budget × 4-fold grid)
  * snn/permutation_test.json          (n_perm=1000 null)
  * hidden_dim_sweep/h*.json + summary (extended to 1024/2048)
  * trained_snn_ensemble/ensemble.json
  * ridge_lag_sweep/summary.json
  * snn_sensitivity/summary.json

Writes:
  * results/cluster/summary.md  — paper-ready table + interpretation prose
  * results/cluster/figures/headline_frontier_multiseed.png  — frontier
    figure with multi-seed error bars from the block_cv grid

Idempotent: missing inputs are logged and skipped (no failure).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("aggregate_cluster_results")


def _read_json(path: Path):
    if not path.is_file():
        logger.warning("missing: %s", path)
        return None
    return json.loads(path.read_text())


def _aggregate_block_cv(rows: list[dict]) -> dict:
    """Group block_cv rows by (model, event_budget, fold), report mean ± std over seeds."""
    by_cell: dict[tuple[str, float, int], list[float]] = {}
    for r in rows:
        key = (r["model"], float(r.get("event_budget", 1.0)), int(r["fold"]))
        by_cell.setdefault(key, []).append(float(r["r2_joint"]))

    summary = {}
    for (model, budget, fold), vs in by_cell.items():
        summary.setdefault((model, budget), []).append({
            "fold": fold,
            "n_seeds": len(vs),
            "mean_r2_joint": float(statistics.mean(vs)),
            "std_r2_joint": float(statistics.stdev(vs)) if len(vs) > 1 else 0.0,
            "best_r2_joint": float(max(vs)),
            "worst_r2_joint": float(min(vs)),
        })

    out: dict[str, dict] = {}
    for (model, budget), folds in summary.items():
        folds = sorted(folds, key=lambda x: x["fold"])
        out.setdefault(model, {})[budget] = {
            "folds": folds,
            "mean_across_folds": float(statistics.mean(f["mean_r2_joint"] for f in folds)),
            "std_across_folds": float(statistics.stdev([f["mean_r2_joint"] for f in folds])) if len(folds) > 1 else 0.0,
        }
    return out


def _plot_frontier_with_errors(block_cv_summary: dict, out_path: Path) -> None:
    """Headline R² vs event budget, one curve per model, with std-across-seeds error bars."""
    style = {
        "ridge": {"color": "#1f77b4", "marker": "o", "label": "Ridge (single-bin counts)"},
        "ridge_lag4": {"color": "#2ca02c", "marker": "^", "label": "Ridge + 4-bin history"},
        "trained_snn": {"color": "#ff7f0e", "marker": "P", "label": "Trained SNN (BPTT)"},
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for model, per_budget in block_cv_summary.items():
        if model not in style:
            continue
        budgets_sorted = sorted(per_budget.keys(), reverse=True)
        means = []
        stds = []
        for b in budgets_sorted:
            cell = per_budget[b]
            means.append(cell["mean_across_folds"])
            stds.append(cell["std_across_folds"])
        ax.errorbar(
            budgets_sorted, means, yerr=stds,
            **style[model], capsize=4, linewidth=2, markersize=8,
        )

    ax.set_xlabel("Event budget  f  (fraction of earliest spike events retained)")
    ax.set_ylabel("Velocity R²  (mean ± std across folds, multi-seed)")
    ax.set_title("Decoder accuracy vs sparse event budget — H100 block CV grid")
    ax.set_xlim(1.05, -0.02)
    ax.set_ylim(-0.05, 0.65)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _format_table(block_cv_summary: dict) -> str:
    """Markdown table: model × event_budget mean ± std (averaged across folds)."""
    budgets = sorted(
        {b for per in block_cv_summary.values() for b in per.keys()}, reverse=True
    )
    lines = ["| Model |" + "".join(f" f={b:.2f} |" for b in budgets)]
    lines.append("|---|" + "---|" * len(budgets))
    for model in ("ridge", "ridge_lag4", "trained_snn"):
        if model not in block_cv_summary:
            continue
        cells = []
        for b in budgets:
            cell = block_cv_summary[model].get(b)
            if cell is None:
                cells.append(" — ")
                continue
            cells.append(f" {cell['mean_across_folds']:+.4f} ± {cell['std_across_folds']:.4f} ")
        lines.append(f"| `{model}` |" + "|".join(cells) + "|")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results/cluster"))
    p.add_argument("--summary-path", type=Path, default=Path("results/cluster/summary.md"))
    p.add_argument(
        "--figure-path",
        type=Path,
        default=Path("results/cluster/figures/headline_frontier_multiseed.png"),
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    md_parts: list[str] = []
    md_parts.append("# Cluster run summary\n")
    md_parts.append(
        "Auto-generated by `scripts/aggregate_cluster_results.py`. Re-run after "
        "every batch of cluster results lands to refresh the table + figure.\n"
    )

    block_cv = _read_json(args.results_dir / "block_cv" / "block_cv.json")
    if block_cv and block_cv.get("rows"):
        summary = _aggregate_block_cv(block_cv["rows"])
        md_parts.append("## Block-CV grid (mean ± std across folds, multi-seed)\n")
        md_parts.append(_format_table(summary))
        md_parts.append("")
        _plot_frontier_with_errors(summary, args.figure_path)

    perm = _read_json(args.results_dir / "snn" / "permutation_test.json")
    if perm and perm.get("rows"):
        md_parts.append("## Permutation test\n")
        for row in perm["rows"]:
            md_parts.append(
                f"- **f = {row['event_budget']:.2f}**: real R² = {row['real_r2_joint']:+.4f}, "
                f"null mean = {row['null_mean']:+.4f}  "
                f"[{row['null_lo_ci']:+.4f}, {row['null_hi_ci']:+.4f}],  "
                f"**p = {row['p_value_one_sided']:.4g}**  (n_perm = {row['n_perm']})"
            )
        md_parts.append("")

    hidden = _read_json(args.results_dir / "hidden_dim_sweep" / "summary.json")
    if hidden and hidden.get("rows"):
        md_parts.append("## Hidden-dim scaling (extended to 1024/2048)\n")
        md_parts.append("| hidden_dim | f=1.0 mean ± std | f=0.25 mean ± std |")
        md_parts.append("|---:|---:|---:|")
        by_h: dict[int, dict[float, dict]] = {}
        for r in hidden["rows"]:
            by_h.setdefault(int(r["hidden_dim"]), {})[float(r["event_budget"])] = r
        for h in sorted(by_h.keys()):
            cells = []
            for b in (1.0, 0.25):
                cell = by_h[h].get(b)
                cells.append(
                    f"{cell['mean_r2_joint']:+.4f} ± {cell['std_r2_joint']:.4f}"
                    if cell else " — "
                )
            md_parts.append(f"| {h} | {cells[0]} | {cells[1]} |")
        md_parts.append("")

    ensemble = _read_json(args.results_dir / "trained_snn_ensemble" / "ensemble.json")
    if ensemble and ensemble.get("results"):
        md_parts.append("## Trained-SNN 10-seed ensemble\n")
        md_parts.append("| f | ensemble R² [95% CI] | individual mean ± std | best individual | ensemble gain |")
        md_parts.append("|---:|---:|---:|---:|---:|")
        for row in ensemble["results"]:
            md_parts.append(
                f"| {row['event_budget']:.2f} | "
                f"{row['ensemble_r2_joint']:+.4f} "
                f"[{row['ensemble_r2_joint_ci_lo']:+.4f}, {row['ensemble_r2_joint_ci_hi']:+.4f}] | "
                f"{row['individual_r2_joint_mean']:+.4f} ± {row['individual_r2_joint_std']:.4f} | "
                f"{row['individual_r2_joint_best']:+.4f} | "
                f"{row['ensemble_gain']:+.4f} |"
            )
        md_parts.append("")

    ridge_lag = _read_json(args.results_dir / "ridge_lag_sweep" / "summary.json")
    if ridge_lag and ridge_lag.get("results"):
        md_parts.append("## Ridge lag-bin sweep\n")
        md_parts.append("| lag | f=1.00 | f=0.50 | f=0.25 | f=0.10 |")
        md_parts.append("|---:|---:|---:|---:|---:|")
        by_lag: dict[int, dict[float, float]] = {}
        for r in ridge_lag["results"]:
            by_lag.setdefault(int(r["lag_bins"]), {})[float(r["event_budget"])] = r["r2_joint"]
        for lag in sorted(by_lag.keys()):
            cells = " | ".join(
                f"{by_lag[lag].get(b, float('nan')):+.4f}" for b in (1.0, 0.5, 0.25, 0.1)
            )
            md_parts.append(f"| {lag} | {cells} |")
        md_parts.append("")

    sens = _read_json(args.results_dir / "snn_sensitivity" / "summary.json")
    if sens and sens.get("results"):
        md_parts.append("## Trained-SNN hyperparameter sensitivity (f=1.0, 3 seeds)\n")
        for knob in ("tau_ms", "threshold", "k_history"):
            md_parts.append(f"### {knob}")
            by_val: dict[float, list[float]] = {}
            for r in sens["results"]:
                if r["knob"] == knob:
                    by_val.setdefault(float(r["value"]), []).append(float(r["r2_joint"]))
            for v in sorted(by_val.keys()):
                vs = by_val[v]
                mn = statistics.mean(vs)
                sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
                md_parts.append(f"- {knob} = {v}: R² = {mn:+.4f} ± {sd:.4f} (n={len(vs)})")
            md_parts.append("")

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text("\n".join(md_parts) + "\n")
    logger.info("wrote %s", args.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
