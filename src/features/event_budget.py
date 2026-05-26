"""Event-budget filtering — keep the earliest fraction f of events in each bin.

This module owns two layers:

  * `apply_event_budget` works directly on raw `(event_times, event_neurons)`
    list pairs. It is what the SNN side imports.
  * `restrict_to_event_budget` is the high-level helper for the ridge
    baseline: it takes a full processed-data dict, filters its events, and
    rebuilds `spike_counts` so the ridge regressor can be re-fit from
    nothing but the retained early events.

Both layers preserve the strict-monotonicity invariant from
`docs/data_interface.md` — `event_times[t]` stays sorted because we only
ever truncate the head of an already-sorted array.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .spike_counts import counts_from_events

logger = logging.getLogger(__name__)


def apply_event_budget(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Keep only the earliest `fraction` of events in each bin.

    Requires `event_times[t]` to be sorted ascending (invariant 2 from
    `docs/data_interface.md`). For a bin with N events, retains the first
    `max(1, floor(fraction * N))` events; empty bins stay empty.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if len(event_times) != len(event_neurons):
        raise ValueError(
            f"event_times and event_neurons length mismatch: "
            f"{len(event_times)} vs {len(event_neurons)}"
        )
    if fraction == 1.0:
        return event_times, event_neurons

    kept_times: list[np.ndarray] = []
    kept_neurons: list[np.ndarray] = []
    for times, neurons in zip(event_times, event_neurons):
        n = times.size
        if n != neurons.size:
            raise ValueError(f"bin has misaligned event arrays: times={n} neurons={neurons.size}")
        if n == 0:
            kept_times.append(times)
            kept_neurons.append(neurons)
            continue
        k = max(1, int(np.floor(fraction * n)))
        kept_times.append(times[:k])
        kept_neurons.append(neurons[:k])
    return kept_times, kept_neurons


def restrict_to_event_budget(
    processed: dict[str, Any],
    fraction: float,
) -> dict[str, Any]:
    """Apply event budget `fraction` to a processed-data dict.

    Returns a new dict (shallow-copied from `processed`) whose
    `event_times`, `event_neurons`, and `spike_counts` reflect only the
    retained early events. All other fields (velocity, splits, metadata)
    are carried over unchanged so the returned dict still satisfies the
    shared contract in `docs/data_interface.md` and can be passed
    straight into the ridge baseline.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    et, en = apply_event_budget(
        processed["event_times"], processed["event_neurons"], fraction
    )
    spike_counts = counts_from_events(en, processed["num_neurons"])

    n_total = int(sum(t.size for t in processed["event_times"]))
    n_kept = int(sum(t.size for t in et))
    pct = 100.0 * n_kept / max(n_total, 1)
    logger.info(
        "restrict_to_event_budget: f=%.2f kept %d/%d events (%.1f%%), spike_counts shape=%s",
        fraction,
        n_kept,
        n_total,
        pct,
        spike_counts.shape,
    )

    out = dict(processed)
    out["event_times"] = et
    out["event_neurons"] = en
    out["spike_counts"] = spike_counts
    return out
