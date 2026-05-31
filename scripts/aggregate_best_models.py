"""Aggregate the best-configuration result of every decoder into one table + figure.

Each model is run at its own optimum (memoryless baselines stay memoryless;
the history-capable decoders use their best-on-val history depth). Reads
whatever result files exist and skips the rest, so it works mid-sweep.

Writes results/best/best_models.{md,json} and results/best/best_frontier.png.
"""
from __future__ import annotations

import csv
import json
import logging
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(message)s")
logger = logging.getLogger("aggregate_best")

REPO = Path(__file__).resolve().parents[1]
BUDGETS = [1.0, 0.5, 0.25, 0.1]

# (display label, kind, path, plot color/marker) — colours match src/evaluation/figstyle.
SOURCES = [
    ("Ridge (counts)",          "json", "results/ridge/ridge_results.json",                "#999999", "o"),
    ("Latency (first-spike)",   "json", "results/latency/latency_results.json",            "#E69F00", "v"),
    ("Ridge + deep history",    "json", "results/ridge/ridge_lag24_results.json",           "#0072B2", "^"),
    ("Reservoir SNN",           "json", "results/snn/snn_results.json",                     "#009E73", "s"),
    ("Trained SNN (deep)",      "csv",  "results/cluster/trained_snn_deep/results.csv",     "#D55E00", "P"),
    ("Deeper SNN (2-layer LIF)", "csv",  "results/cluster/deeper_snn_deep/results.csv",      "#CC79A7", "D"),
]


def _rows(kind: str, path: Path):
    if not path.exists():
        return None
    if kind == "json":
        return json.loads(path.read_text())["results"]
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _by_budget(rows):
    agg: dict[float, list[float]] = {}
    for r in rows:
        f = round(float(r["event_budget"]), 2)
        agg.setdefault(f, []).append(float(r["r2_joint"]))
    return {f: (st.mean(v), st.pstdev(v)) for f, v in agg.items()}


def main() -> int:
    from src.evaluation.figstyle import apply_style
    apply_style()
    out_dir = REPO / "results" / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    for label, kind, rel, color, marker in SOURCES:
        rows = _rows(kind, REPO / rel)
        if not rows:
            logger.info("skip %s (no file at %s)", label, rel)
            continue
        stats = _by_budget(rows)
        table[label] = {f"f={f}": {"mean": m, "std": s} for f, (m, s) in sorted(stats.items(), reverse=True)}
        xs = [f for f in BUDGETS if f in stats]
        ax.errorbar(xs, [stats[f][0] for f in xs], yerr=[stats[f][1] for f in xs],
                    color=color, marker=marker, label=label, capsize=3, linewidth=2, markersize=7)
        logger.info("%-24s %s", label, "  ".join(f"f={f}:{stats[f][0]:+.3f}" for f in xs))

    ax.set_xlabel("Event budget  $f$  (fraction of earliest spikes kept)")
    ax.set_ylabel("Velocity decode  $R^2$  (mean ± std across seeds)")
    ax.set_xlim(1.05, 0.03); ax.set_xticks([1.0, 0.75, 0.5, 0.25, 0.1])
    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "best_frontier.png")
    fig.savefig(out_dir / "best_frontier.pdf")
    plt.close(fig)

    (out_dir / "best_models.json").write_text(json.dumps(table, indent=2))
    # Markdown table
    lines = ["| Decoder (best config) | " + " | ".join(f"f={f:.2f}" for f in BUDGETS) + " |",
             "|---|" + "---|" * len(BUDGETS)]
    for label, cells in table.items():
        row = [label]
        for f in BUDGETS:
            c = cells.get(f"f={f}")
            row.append(f"{c['mean']:+.3f} ± {c['std']:.3f}" if c else " — ")
        lines.append("| " + " | ".join(row) + " |")
    (out_dir / "best_models.md").write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", out_dir / "best_models.md")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
