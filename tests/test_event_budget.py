"""Event-budget filter invariants.

These cover the contract `restrict_to_event_budget` is held to by both
the ridge baseline and (eventually) the SNN side.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.features.event_budget import apply_event_budget, restrict_to_event_budget


def test_fraction_one_is_identity(mock_data):
    """f=1.0 returns the input unchanged."""
    et, en = apply_event_budget(
        mock_data["event_times"], mock_data["event_neurons"], fraction=1.0
    )
    assert et is mock_data["event_times"]
    assert en is mock_data["event_neurons"]


def test_filter_keeps_earliest_events(mock_data):
    """At f=0.25, every retained event is at or before the f-th original event."""
    et, en = apply_event_budget(
        mock_data["event_times"], mock_data["event_neurons"], fraction=0.25
    )
    for t in range(len(et)):
        n_orig = mock_data["event_times"][t].size
        k = max(1, int(np.floor(0.25 * n_orig))) if n_orig else 0
        assert et[t].size == k
        if k:
            np.testing.assert_array_equal(et[t], mock_data["event_times"][t][:k])
            np.testing.assert_array_equal(en[t], mock_data["event_neurons"][t][:k])


def test_restrict_rebuilds_spike_counts(mock_data):
    """restrict_to_event_budget must update spike_counts to match the kept events."""
    sub = restrict_to_event_budget(mock_data, fraction=0.5)
    num_neurons = sub["spike_counts"].shape[1]
    sample = np.linspace(0, sub["spike_counts"].shape[0] - 1, num=20).astype(int)
    for t in sample:
        expected = np.bincount(sub["event_neurons"][t], minlength=num_neurons)
        np.testing.assert_array_equal(sub["spike_counts"][t], expected)


def test_restrict_preserves_velocity_and_splits(mock_data):
    """Event-budget filtering must never touch velocity, splits, or metadata."""
    sub = restrict_to_event_budget(mock_data, fraction=0.25)
    np.testing.assert_array_equal(sub["velocity"], mock_data["velocity"])
    np.testing.assert_array_equal(sub["train_idx"], mock_data["train_idx"])
    np.testing.assert_array_equal(sub["val_idx"], mock_data["val_idx"])
    np.testing.assert_array_equal(sub["test_idx"], mock_data["test_idx"])
    assert sub["bin_size_ms"] == mock_data["bin_size_ms"]


@pytest.mark.parametrize("bad_fraction", [-0.1, 0.0, 1.1, 2.0])
def test_invalid_fraction_raises(mock_data, bad_fraction):
    """fraction must be in (0, 1]."""
    with pytest.raises(ValueError):
        apply_event_budget(
            mock_data["event_times"], mock_data["event_neurons"], fraction=bad_fraction
        )
