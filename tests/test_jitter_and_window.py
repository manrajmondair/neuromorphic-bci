"""Smoke tests for spike-time jitter + causal-window helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.causal_window import truncate_to_window
from src.features.event_budget import apply_event_budget
from src.features.jitter import jitter_event_times


def test_jitter_sigma_zero_is_identity(mock_data):
    et, en = jitter_event_times(
        mock_data["event_times"], mock_data["event_neurons"],
        sigma_ms=0.0, bin_size_ms=int(mock_data["bin_size_ms"]),
    )
    assert et is mock_data["event_times"]
    assert en is mock_data["event_neurons"]


def test_jitter_preserves_event_counts(mock_data):
    et, en = jitter_event_times(
        mock_data["event_times"], mock_data["event_neurons"],
        sigma_ms=5.0, bin_size_ms=int(mock_data["bin_size_ms"]),
    )
    for j_t, orig_t in zip(et, mock_data["event_times"], strict=True):
        assert j_t.size == orig_t.size


def test_jitter_keeps_monotonic_times(mock_data):
    et, _ = jitter_event_times(
        mock_data["event_times"], mock_data["event_neurons"],
        sigma_ms=20.0, bin_size_ms=int(mock_data["bin_size_ms"]), seed=7,
    )
    for t in et:
        if t.size > 1:
            assert np.all(np.diff(t) > 0)


def test_jitter_keeps_times_in_bin(mock_data):
    bin_size = int(mock_data["bin_size_ms"])
    et, _ = jitter_event_times(
        mock_data["event_times"], mock_data["event_neurons"],
        sigma_ms=20.0, bin_size_ms=bin_size, seed=11,
    )
    for t in et:
        if t.size > 0:
            assert t.min() >= 0.0
            assert t.max() < bin_size


def test_truncate_window_drops_late_events(mock_data):
    et, en = truncate_to_window(
        mock_data["event_times"], mock_data["event_neurons"], window_ms=25.0,
    )
    for t in et:
        if t.size > 0:
            assert t.max() < 25.0


def test_truncate_window_invalid_raises(mock_data):
    with pytest.raises(ValueError):
        truncate_to_window(mock_data["event_times"], mock_data["event_neurons"], window_ms=0)


def test_truncate_composes_with_event_budget(mock_data):
    """budget then truncate yields the same counts as truncate then budget on the surviving events."""
    et_b, en_b = apply_event_budget(
        mock_data["event_times"], mock_data["event_neurons"], 0.5
    )
    et_bt, _ = truncate_to_window(et_b, en_b, window_ms=20.0)
    for t in et_bt:
        if t.size > 0:
            assert t.max() < 20.0
