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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(message)s")
logger = logging.getLogger("aggregate_best")

REPO = Path(__file__).resolve().parents[1]
BUDGETS = [1.0, 0.5, 0.25, 0.1]

# (display label, kind, path, plot color/marker)
SOURCES = [
    ("Ridge (counts)",          "json", "results/ridge/ridge_results.json",                "#1f77b4", "o"),
    ("Latency (first-spike)",   "json", "results/latency/latency_results.json",            "#8c564b", "v"),
    ("Ridge + deep history",    "json", "results/ridge/ridge_lag20_results.json",           "#2ca02c", "^"),
    ("Reservoir SNN",           "json", "results/snn/snn_results.json",                     "#9467bd", "s"),
    ("Trained SNN (deep)",      "csv",  "results/cluster/trained_snn_deep/results.csv",     "#ff7f0e", "P"),
    ("Deeper SNN (recurrent)",  "csv",  "results/cluster/deeper_snn_deep/results.csv",      "#d62728", "D"),
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
    out_dir = REPO / "results" / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(8, 5.5))

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

    ax.set_xlabel("Event budget  f  (fraction of earliest spike events kept)")
    ax.set_ylabel("Velocity R²  (mean ± std across seeds)")
    ax.set_title("Best configuration of every decoder vs sparse event budget")
    ax.set_xlim(1.05, -0.02); ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "best_frontier.png", dpi=300, bbox_inches="tight")
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
