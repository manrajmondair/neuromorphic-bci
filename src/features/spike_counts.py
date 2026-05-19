"""Spike-count feature construction for the ridge baseline.

Owned by data-ridge-baseline.
"""
from __future__ import annotations

import numpy as np


def counts_from_events(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    num_neurons: int,
) -> np.ndarray:
    """Rebuild [num_bins, num_neurons] spike counts from per-bin event lists.

    Used after event-budget filtering, where event_times/event_neurons have
    been truncated so the cached `spike_counts` array no longer matches.
    """
    num_bins = len(event_times)
    counts = np.zeros((num_bins, num_neurons), dtype=np.int32)
    for t in range(num_bins):
        neurons_t = event_neurons[t]
        if neurons_t.size == 0:
            continue
        np.add.at(counts[t], neurons_t, 1)
    return counts
