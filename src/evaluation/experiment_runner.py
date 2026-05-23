"""Shared experiment-result persistence.

Every model writes one row per (event_budget, seed) to two places:

  * `results/{model}/results.csv` — flat CSV for quick diffing.
  * `results/{model}/{model}_results.json` — the canonical tracking file
    the plotting code reads. The JSON is the shared analysis format both
    sides commit to so curves can be overlaid without conversion.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULT_COLUMNS: tuple[str, ...] = (
    "model",
    "event_budget",
    "seed",
    "r2_vx",
    "r2_vy",
    "r2_joint",
    "n_events_used",
    "n_events_total",
    "notes",
)


def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one experiment row to the model's results.csv (creates with header)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def save_json_results(
    json_path: Path,
    model: str,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    metric: str = "velocity_r2",
    dataset: str = "NLB_MC_RTT",
) -> None:
    """Write the canonical per-model JSON tracking file.

    Schema:
        {
          "model": str,
          "metric": "velocity_r2",
          "dataset": "NLB_MC_RTT",
          "config": { ... arbitrary run config ... },
          "results": [ {event_budget, seed, r2_vx, r2_vy, r2_joint, ...}, ... ]
        }
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {
        "model": model,
        "metric": metric,
        "dataset": dataset,
        "config": config,
        "results": rows,
    }
    with json_path.open("w") as f:
        json.dump(blob, f, indent=2, sort_keys=False)
    logger.info("wrote %s (%d rows)", json_path, len(rows))
