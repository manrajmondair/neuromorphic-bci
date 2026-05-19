"""Mock processed-data generator with the exact schema in docs/data_interface.md.

Lets snn-latency-decoder develop end-to-end before real MC_RTT preprocessing
lands. When the real .npz is ready, swap this import for
`src.data.preprocess.load_processed` — same dict shape.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def make_mock_processed_data(
    num_bins: int = 10000,
    num_neurons: int = 96,
    bin_size_ms: int = 50,
    mean_rate_hz: float = 8.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a processed-data dict matching the contract.

    Spike events per (bin, neuron) are Poisson-sampled with `mean_rate_hz`,
    velocity is a smooth random walk so a linear decoder has something real
    to fit, and bin events are sorted ascending — invariant 2 from the doc.
    """
    rng = np.random.default_rng(seed)
    bin_size_s = bin_size_ms / 1000.0
    lam = mean_rate_hz * bin_size_s

    spike_counts = rng.poisson(lam=lam, size=(num_bins, num_neurons)).astype(np.int32)

    event_times: list[np.ndarray] = []
    event_neurons: list[np.ndarray] = []
    for t in range(num_bins):
        counts_t = spike_counts[t]
        total = int(counts_t.sum())
        if total == 0:
            event_times.append(np.zeros(0, dtype=np.float32))
            event_neurons.append(np.zeros(0, dtype=np.int32))
            continue
        neurons_t = np.repeat(np.arange(num_neurons, dtype=np.int32), counts_t)
        times_t = rng.uniform(0.0, bin_size_ms, size=total).astype(np.float32)
        order = np.argsort(times_t, kind="stable")
        event_times.append(times_t[order])
        event_neurons.append(neurons_t[order])

    steps = rng.normal(scale=0.05, size=(num_bins, 2)).astype(np.float32)
    velocity = np.cumsum(steps, axis=0)
    velocity -= velocity.mean(axis=0, keepdims=True)

    n_train = int(0.70 * num_bins)
    n_val = int(0.15 * num_bins)
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, num_bins, dtype=np.int64)

    return {
        "spike_counts": spike_counts,
        "event_times": event_times,
        "event_neurons": event_neurons,
        "velocity": velocity,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "bin_size_ms": bin_size_ms,
        "num_neurons": num_neurons,
        "dataset_name": "MOCK_MC_RTT",
    }
