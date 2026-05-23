"""Secondary efficiency analysis for the ridge baseline.

The proposal commits to reporting implementation-relevant quantities
alongside decoding R²: how many spike events the decoder actually
processes at each event budget and how many dense MAC operations an
event-driven architecture would avoid relative to the dense ridge
baseline. None of this is a primary decoding metric — it justifies the
neuromorphic motivation in the final write-up.

What this module computes:

  1. Per event budget f ∈ {1.00, 0.50, 0.25, 0.10}, the distribution of
     retained spike events per 50 ms bin (total, mean, median, max, std).

  2. Dense MACs the ridge baseline performs per prediction. Ridge
     prediction is ŷ = X @ W + b with X ∈ R^(1 × N) and W ∈ R^(N × 2),
     so each prediction costs 2 · N multiply-accumulates. Across the
     whole recording that's 2 · N · num_bins.

  3. The equivalent event-driven cost. An event-driven implementation of
     the same linear readout adds W[neuron, :] (2 MACs) into a running
     accumulator on every retained spike, then emits one prediction per
     bin (2 bias adds). Total: 2 · events_total + 2 · num_bins.

  4. MACs avoided = dense − event-driven, plus the fraction avoided. This
     is the headline secondary number for the paper.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from src.features.event_budget import apply_event_budget

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Operation-count primitives
# -------------------------------------------------------------------------


def dense_macs_per_prediction(num_neurons: int, num_outputs: int = 2) -> int:
    """MACs for one dense ridge prediction: ŷ = X @ W + b on X ∈ R^(1 × N)."""
    if num_neurons <= 0 or num_outputs <= 0:
        raise ValueError(f"num_neurons and num_outputs must be positive")
    return int(num_neurons * num_outputs)


def dense_macs_total(num_neurons: int, num_bins: int, num_outputs: int = 2) -> int:
    """Total dense MACs across the whole recording."""
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    return int(dense_macs_per_prediction(num_neurons, num_outputs) * num_bins)


def event_driven_macs_total(
    events_total: int,
    num_bins: int,
    num_outputs: int = 2,
) -> int:
    """Total MACs for an event-driven linear readout.

    Per retained spike: `num_outputs` MACs to accumulate W[neuron, :] into
    the running output. Per bin emission: `num_outputs` MACs for the bias
    add and accumulator reset. Reads of the accumulator are not counted
    as MACs (they're not multiply-accumulates).
    """
    if events_total < 0 or num_bins < 0:
        raise ValueError("events_total and num_bins must be non-negative")
    return int(num_outputs * events_total + num_outputs * num_bins)


def macs_avoided(dense: int, event_driven: int) -> dict[str, float]:
    """Dict of (dense, event-driven, avoided, avoided-fraction)."""
    avoided = max(int(dense) - int(event_driven), 0)
    fraction = float(avoided) / float(dense) if dense > 0 else 0.0
    return {
        "dense_macs": int(dense),
        "event_driven_macs": int(event_driven),
        "macs_avoided": int(avoided),
        "macs_avoided_fraction": fraction,
    }


# -------------------------------------------------------------------------
# Spike-event accounting
# -------------------------------------------------------------------------


def events_per_bin_stats(event_times: list[np.ndarray]) -> dict[str, Any]:
    """Summary statistics for the per-bin event-count distribution."""
    if not event_times:
        return {
            "events_total": 0,
            "num_bins": 0,
            "events_per_bin_mean": 0.0,
            "events_per_bin_median": 0.0,
            "events_per_bin_std": 0.0,
            "events_per_bin_min": 0,
            "events_per_bin_max": 0,
            "empty_bin_fraction": 0.0,
        }
    counts = np.array([int(t.size) for t in event_times], dtype=np.int64)
    return {
        "events_total": int(counts.sum()),
        "num_bins": int(counts.size),
        "events_per_bin_mean": float(counts.mean()),
        "events_per_bin_median": float(np.median(counts)),
        "events_per_bin_std": float(counts.std(ddof=0)),
        "events_per_bin_min": int(counts.min()),
        "events_per_bin_max": int(counts.max()),
        "empty_bin_fraction": float((counts == 0).mean()),
    }


# -------------------------------------------------------------------------
# Top-level analyses
# -------------------------------------------------------------------------


def profile_event_budget(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    fraction: float,
    num_neurons: int,
    num_outputs: int = 2,
) -> dict[str, Any]:
    """Run the event-count and MAC-avoidance analysis for one budget level.

    Lightweight on purpose: only counts events and arithmetic — no model
    training. Use this to fold a single budget into a comparison table.
    """
    if fraction < 1.0:
        et, _ = apply_event_budget(event_times, event_neurons, fraction)
    else:
        et = event_times

    spike_stats = events_per_bin_stats(et)
    num_bins = spike_stats["num_bins"]
    dense_t = dense_macs_total(num_neurons, num_bins, num_outputs)
    event_driven_t = event_driven_macs_total(spike_stats["events_total"], num_bins, num_outputs)
    op_stats = macs_avoided(dense_t, event_driven_t)

    return {
        "event_budget": float(fraction),
        **spike_stats,
        **op_stats,
    }


def compute_efficiency_summary(
    processed: dict[str, Any],
    fractions: Sequence[float] = (1.00, 0.50, 0.25, 0.10),
    num_outputs: int = 2,
    model: str = "ridge",
) -> dict[str, Any]:
    """Build the full computational-efficiency summary blob.

    The returned dict serializes 1:1 into the secondary tracking file
    `results/ridge/computational_efficiency.json`.
    """
    event_times = processed["event_times"]
    event_neurons = processed["event_neurons"]
    num_neurons = int(processed["num_neurons"])
    num_bins = len(event_times)

    macs_per_pred = dense_macs_per_prediction(num_neurons, num_outputs)
    dense_t = dense_macs_total(num_neurons, num_bins, num_outputs)

    logger.info(
        "efficiency analysis: %d bins, %d neurons, %d outputs, dense_macs/pred=%d, dense_macs_total=%d",
        num_bins,
        num_neurons,
        num_outputs,
        macs_per_pred,
        dense_t,
    )

    budget_rows: list[dict[str, Any]] = []
    for f in fractions:
        row = profile_event_budget(
            event_times, event_neurons, f, num_neurons, num_outputs
        )
        budget_rows.append(row)
        logger.info(
            "budget f=%.2f: events_total=%d (mean=%.2f/bin) | dense=%d event_driven=%d avoided=%d (%.1f%%)",
            f,
            row["events_total"],
            row["events_per_bin_mean"],
            row["dense_macs"],
            row["event_driven_macs"],
            row["macs_avoided"],
            100.0 * row["macs_avoided_fraction"],
        )

    return {
        "model": model,
        "dataset": str(processed.get("dataset_name", "NLB_MC_RTT")),
        "bin_size_ms": int(processed.get("bin_size_ms", 50)),
        "num_neurons": num_neurons,
        "num_outputs": int(num_outputs),
        "num_bins": int(num_bins),
        "macs_per_prediction_dense": int(macs_per_pred),
        "dense_macs_total": int(dense_t),
        "budgets": budget_rows,
        "notes": (
            "Dense MAC count = 2·N·T (X @ W + b for X[1,N], W[N,2], T bins). "
            "Event-driven MAC count = 2·E + 2·T (per-spike accumulator update "
            "plus per-bin bias). MACs avoided = dense − event_driven; this is "
            "the secondary efficiency number reported alongside R²."
        ),
    }


def save_efficiency_json(summary: dict[str, Any], path) -> None:
    """Persist the efficiency summary to JSON."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    logger.info("wrote %s", path)
