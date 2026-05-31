"""Plot hidden-dim scaling curve from the run_hidden_dim_sweep summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from src.evaluation.figstyle import apply_style, save_fig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path,
                   default=Path("results/hidden_dim_sweep/summary.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/figures/supp_hidden_dim_scaling.png"))
    args = p.parse_args()
    apply_style()

    data = json.loads(args.summary.read_text())
    by_budget: dict[float, list[dict]] = {}
    for r in data["rows"]:
        by_budget.setdefault(float(r["event_budget"]), []).append(r)
    for budget in by_budget:
        by_budget[budget].sort(key=lambda r: r["hidden_dim"])

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    budgets = sorted(by_budget, reverse=True)
    shades = cm.viridis(np.linspace(0.15, 0.85, len(budgets)))
    for budget, c in zip(budgets, shades):
        rows = by_budget[budget]
        xs = [r["hidden_dim"] for r in rows]
        ys = np.array([r["mean_r2_joint"] for r in rows])
        es = np.array([r["std_r2_joint"] for r in rows])
        ax.fill_between(xs, ys - es, ys + es, color=c, alpha=0.15)
        ax.plot(xs, ys, marker="o", color=c, label=f"$f$ = {budget:.2f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Hidden-layer width  (LIF units)")
    ax.set_ylabel("Velocity decode  $R^2$")
    ax.grid(True, which="both")
    ax.legend(loc="center right", title="event budget")
    fig.tight_layout()
    save_fig(fig, args.out.with_suffix(""))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
