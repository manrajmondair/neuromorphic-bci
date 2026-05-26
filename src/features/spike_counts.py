"""Spike-count feature construction for the ridge baseline.

Single canonical function for going from per-bin sparse event lists back
to a dense `[num_bins, num_neurons]` count matrix. Used both inside
`apply_event_budget`'s downstream step and standalone when only counts
need to be reconstructed.

Owned by data-ridge-baseline.
"""
from __future__ import annotations

import numpy as np


def counts_from_events(
    event_neurons: list[np.ndarray],
    num_neurons: int,
) -> np.ndarray:
    """Rebuild a dense `[num_bins, num_neurons]` spike-count matrix.

    For each bin t and each neuron id in `event_neurons[t]`, the
    corresponding count is incremented by one. Each entry in
    `event_neurons[t]` is one retained spike (post event-budget filtering
    if applicable).
    """
    num_bins = len(event_neurons)
    counts = np.zeros((num_bins, num_neurons), dtype=np.int32)
    for t in range(num_bins):
        neurons_t = event_neurons[t]
        if neurons_t.size == 0:
            continue
        np.add.at(counts[t], neurons_t, 1)
    return counts
