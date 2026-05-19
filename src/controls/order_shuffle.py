"""Order-shuffle control: same retained events, within-bin order permuted.

Owned by snn-latency-decoder.
"""
from __future__ import annotations

import numpy as np


def shuffle_within_bin_order(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Permute the per-bin order of (time, neuron) pairs. Counts are preserved."""
    rng = np.random.default_rng(seed)
    shuffled_times: list[np.ndarray] = []
    shuffled_neurons: list[np.ndarray] = []
    for times, neurons in zip(event_times, event_neurons):
        n = times.size
        if n <= 1:
            shuffled_times.append(times)
            shuffled_neurons.append(neurons)
            continue
        perm = rng.permutation(n)
        shuffled_times.append(times[perm])
        shuffled_neurons.append(neurons[perm])
    return shuffled_times, shuffled_neurons
