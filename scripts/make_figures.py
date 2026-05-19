"""Combine all model result CSVs and emit the headline R² vs. event-budget figure."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.plots import plot_r2_vs_event_budget


def main() -> None:
    parts = []
    for csv in [Path("results/ridge/results.csv"), Path("results/snn/results.csv")]:
        if csv.exists():
            parts.append(pd.read_csv(csv))
    if not parts:
        raise SystemExit("no result CSVs found — run run_ridge.py and run_snn.py first")
    combined = pd.concat(parts, ignore_index=True)
    combined_path = Path("results/figures/comparison.csv")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    plot_r2_vs_event_budget(combined_path, Path("results/figures/r2_vs_event_budget.png"))
    print("wrote results/figures/comparison.csv and r2_vs_event_budget.png")


if __name__ == "__main__":
    main()
