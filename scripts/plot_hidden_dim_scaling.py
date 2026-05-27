"""Plot hidden-dim scaling curve from the run_hidden_dim_sweep summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path,
                   default=Path("results/hidden_dim_sweep/summary.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/figures/hidden_dim_scaling.png"))
    args = p.parse_args()

    data = json.loads(args.summary.read_text())
    by_budget: dict[float, list[dict]] = {}
    for r in data["rows"]:
        by_budget.setdefault(float(r["event_budget"]), []).append(r)
    for budget in by_budget:
        by_budget[budget].sort(key=lambda r: r["hidden_dim"])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for budget in sorted(by_budget, reverse=True):
        rows = by_budget[budget]
        xs = [r["hidden_dim"] for r in rows]
        ys = [r["mean_r2_joint"] for r in rows]
        es = [r["std_r2_joint"] for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, linewidth=2,
                    markersize=7, label=f"f = {budget:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Hidden-layer width (LIF units)")
    ax.set_ylabel("Velocity R² (joint, mean ± std over seeds)")
    ax.set_title("Trained SNN hidden-dim scaling curve (k_history = 4)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
