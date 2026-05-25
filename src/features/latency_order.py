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
    """Return [num_bins, num_neurons] of within-bin first-spike latency in ms.

    Neurons with no spike in the bin get `bin_size_ms` (treated as "never").
    """
    num_bins = len(event_times)
    taus = np.full((num_bins, num_neurons), bin_size_ms, dtype=np.float32)

    for i in range(len(event_times)):
        times = event_times[i]
        neurons = event_neurons[i]
        for j in range(len(neurons)):
            neuron = neurons[j]
            time = times[j]
            if taus[i, neuron] == bin_size_ms:
                taus[i, neuron] = time
            
    return taus


def pairwise_order_features(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    pairs: np.ndarray,
    num_neurons: int,
) -> np.ndarray:
    """For neuron pairs (i, j), 1 if i fires before j in the bin, else 0.

    `pairs` has shape [P, 2]. Returns [num_bins, P].
    """

    comparisons = np.zero((num_bins, num_neurons), dtype=np.float32)
    
    raise NotImplementedError
