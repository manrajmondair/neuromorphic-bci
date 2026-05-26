"""Spike-latency and spike-order features for the SNN decoder.

Owned by snn-latency-decoder.
"""
from __future__ import annotations

import numpy as np


def time_to_first_spike(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    num_neurons: int,
    bin_size_ms: int,
) -> np.ndarray:
    """Return `[num_bins, num_neurons]` of within-bin first-spike latencies in ms.

    Neurons that did not fire in bin `t` get `bin_size_ms` as a sentinel ("never
    fired"). Assumes `event_times[t]` is sorted ascending — invariant 2 from
    `docs/data_interface.md` — so the first appearance of each neuron is its
    earliest spike.
    """
    num_bins = len(event_times)
    taus = np.full((num_bins, num_neurons), float(bin_size_ms), dtype=np.float32)
    for t in range(num_bins):
        times = event_times[t]
        neurons = event_neurons[t]
        if neurons.size == 0:
            continue
        # The first index of each unique neuron under stable ordering is its earliest spike.
        seen = np.zeros(num_neurons, dtype=bool)
        for k in range(neurons.size):
            n = int(neurons[k])
            if not seen[n]:
                taus[t, n] = float(times[k])
                seen[n] = True
    return taus


def pairwise_order_features(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    pairs: np.ndarray,
    num_neurons: int,
    bin_size_ms: int,
) -> np.ndarray:
    """For each neuron pair `(i, j)`, 1 if `i` fires before `j` in the bin, else 0.

    `pairs` has shape `[P, 2]`. Returns `[num_bins, P]`. Bins where neither
    neuron fired (both at the `bin_size_ms` sentinel) collapse to 0 because
    `tau_i < tau_j` is false on equal sentinels.
    """
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must be [P, 2], got shape {pairs.shape}")
    taus = time_to_first_spike(event_times, event_neurons, num_neurons, bin_size_ms)
    i = pairs[:, 0]
    j = pairs[:, 1]
    return (taus[:, i] < taus[:, j]).astype(np.float32)
