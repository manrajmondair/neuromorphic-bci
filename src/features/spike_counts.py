"""Spike-count feature construction for the ridge baseline.

Single canonical function for going from per-bin sparse event lists back
to a dense `[num_bins, num_neurons]` count matrix, plus a lag-feature
helper that concatenates the current bin with the previous `k` bins
along the feature axis without leaking across train/val/test boundaries.

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


def stack_lag_features(
    spike_counts: np.ndarray,
    num_lags: int,
    split_starts: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Concatenate `spike_counts[t]` with the previous `num_lags` bins.

    Returns `[num_bins, num_neurons * (num_lags + 1)]` where each row is
    `[counts[t], counts[t-1], counts[t-2], ..., counts[t-num_lags]]`.

    To avoid feature leakage across split boundaries (a val-bin pulling
    spikes from a train bin via its lag features), pass `split_starts`
    as the sorted first bin index of every split. For each bin `t` that
    falls within `num_lags` of the start of its own split, the lag
    columns that would reach across the boundary are zeroed.

    For `num_lags=0` this returns `spike_counts` unchanged (as float32).
    """
    if num_lags < 0:
        raise ValueError(f"num_lags must be >= 0, got {num_lags}")
    if spike_counts.ndim != 2:
        raise ValueError(f"spike_counts must be 2-D, got {spike_counts.shape}")

    counts = spike_counts.astype(np.float32, copy=False)
    if num_lags == 0:
        return counts

    num_bins, num_neurons = counts.shape
    out = np.zeros((num_bins, num_neurons * (num_lags + 1)), dtype=np.float32)
    out[:, :num_neurons] = counts
    for k in range(1, num_lags + 1):
        out[k:, k * num_neurons : (k + 1) * num_neurons] = counts[:-k]

    if split_starts is not None:
        # Zero any lag columns that would reach across a split boundary.
        sorted_starts = sorted(set(int(s) for s in split_starts))
        for s in sorted_starts:
            if s <= 0:
                continue
            # Bins [s, s + num_lags) have at least one lag column that
            # reaches back across the boundary at `s`.
            end = min(num_bins, s + num_lags)
            for t in range(s, end):
                reaches_back = num_lags - (t - s)
                # Zero the last `reaches_back` lag blocks for this row.
                # Block k is columns [k*num_neurons : (k+1)*num_neurons].
                first_bad_k = num_lags + 1 - reaches_back
                out[t, first_bad_k * num_neurons :] = 0.0

    return out
