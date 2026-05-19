"""Final-figure helpers. Owned jointly; lives on main."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_r2_vs_event_budget(results_csv: Path, out_path: Path) -> None:
    """Read combined results.csv and draw the headline R² vs. event-budget curve."""
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, sub in df.groupby("model"):
        agg = sub.groupby("event_budget")["r2_mean"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["event_budget"], agg["mean"], yerr=agg["std"], label=model, marker="o")
    ax.set_xlabel("Event budget f")
    ax.set_ylabel("Velocity R² (mean of vx, vy)")
    ax.set_title("Decoding accuracy vs. sparse event budget")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
