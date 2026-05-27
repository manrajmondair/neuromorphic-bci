"""Causal within-bin truncation — keep only events in the first `window_ms` of each bin.

For a deployed BCI the decoder must emit a prediction *during* the bin
rather than after it. This helper lets us study how decoding accuracy
falls as the window shrinks.

Operates on (event_times, event_neurons) sparse lists, so it composes
freely with the event-budget filter from src/features/event_budget.py.
"""
from __future__ import annotations

import numpy as np


def truncate_to_window(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    window_ms: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Drop every event with time >= window_ms in each bin."""
    if window_ms <= 0:
        raise ValueError(f"window_ms must be positive, got {window_ms}")
    out_t: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    for times, neurons in zip(event_times, event_neurons, strict=True):
        if times.size == 0:
            out_t.append(times)
            out_n.append(neurons)
            continue
        keep = times < window_ms
        out_t.append(times[keep])
        out_n.append(neurons[keep])
    return out_t, out_n
