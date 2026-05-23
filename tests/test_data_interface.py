"""Schema invariants from docs/data_interface.md.

If any of these fail, the contract every model in the project relies on
has drifted and ridge / SNN / shuffle results stop being comparable.
"""
from __future__ import annotations

import numpy as np


def test_mock_data_has_required_keys(mock_data):
    expected = {
        "spike_counts",
        "event_times",
        "event_neurons",
        "velocity",
        "train_idx",
        "val_idx",
        "test_idx",
        "bin_size_ms",
        "num_neurons",
        "dataset_name",
    }
    assert expected.issubset(mock_data.keys())


def test_event_lists_align_with_spike_counts(mock_data):
    """spike_counts[t, n] must equal the number of times neuron n fired in event_neurons[t]."""
    spike_counts = mock_data["spike_counts"]
    event_neurons = mock_data["event_neurons"]
    num_bins, num_neurons = spike_counts.shape
    sample = np.linspace(0, num_bins - 1, num=30).astype(int)
    for t in sample:
        rebuilt = np.bincount(event_neurons[t], minlength=num_neurons)
        np.testing.assert_array_equal(spike_counts[t], rebuilt)


def test_event_times_monotonic(mock_data):
    """event_times[t] must be sorted ascending — invariant 2 from data_interface.md."""
    event_times = mock_data["event_times"]
    for t, times in enumerate(event_times):
        if times.size > 1:
            assert np.all(np.diff(times) >= 0), f"bin {t}: event_times not monotonic"


def test_splits_disjoint_and_in_range(mock_data):
    """train/val/test indices must be disjoint and within [0, num_bins)."""
    num_bins = mock_data["spike_counts"].shape[0]
    union = np.concatenate(
        [mock_data["train_idx"], mock_data["val_idx"], mock_data["test_idx"]]
    )
    assert union.min() >= 0
    assert union.max() < num_bins
    assert len(set(union.tolist())) == union.size, "split indices overlap"


def test_velocity_shape_and_finite(mock_data):
    """velocity must be [num_bins, 2] and contain no NaN/Inf."""
    velocity = mock_data["velocity"]
    num_bins = mock_data["spike_counts"].shape[0]
    assert velocity.shape == (num_bins, 2)
    assert np.isfinite(velocity).all()
