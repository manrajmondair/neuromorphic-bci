"""Shared experiment runner — writes one row per (model, budget, seed) to CSV."""
from __future__ import annotations

import csv
from pathlib import Path

RESULT_COLUMNS = [
    "model",
    "event_budget",
    "seed",
    "r2_vx",
    "r2_vy",
    "r2_mean",
    "n_events_used",
    "n_events_total",
    "notes",
]


def append_result(csv_path: Path, row: dict) -> None:
    """Append one experiment row to the model's results.csv (creates with header)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})
