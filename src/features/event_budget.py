"""Event-budget filtering — keep earliest fraction f of events in each bin.

Shared by both branches. Both ridge and SNN read events through this.
"""
from __future__ import annotations

import numpy as np


def apply_event_budget(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Keep only the earliest `fraction` of events in each bin.

    Assumes event_times[t] is sorted ascending (invariant from data interface).
    """
    assert 0.0 < fraction <= 1.0, f"fraction must be in (0, 1], got {fraction}"
    if fraction == 1.0:
        return event_times, event_neurons

    kept_times: list[np.ndarray] = []
    kept_neurons: list[np.ndarray] = []
    for times, neurons in zip(event_times, event_neurons):
        n = times.size
        if n == 0:
            kept_times.append(times)
            kept_neurons.append(neurons)
            continue
        k = max(1, int(np.floor(fraction * n)))
        kept_times.append(times[:k])
        kept_neurons.append(neurons[:k])
    return kept_times, kept_neurons
