"""Efficiency accounting — the secondary "MACs avoided" headline.

We don't try to verify the exact MAC count of a particular hardware
implementation. We do verify that the arithmetic in efficiency_tracker.py
is internally consistent and matches the formulas documented at the top
of that module.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from src.evaluation.efficiency_tracker import (
    compute_efficiency_summary,
    dense_macs_per_prediction,
    dense_macs_total,
    event_driven_macs_total,
    macs_avoided,
    profile_event_budget,
)


def test_dense_macs_formula():
    assert dense_macs_per_prediction(num_neurons=98, num_outputs=2) == 196
    assert dense_macs_total(num_neurons=98, num_bins=1000, num_outputs=2) == 196_000


def test_event_driven_macs_formula():
    # 2 MACs per spike + 2 MACs per bin (bias add)
    assert event_driven_macs_total(events_total=5, num_bins=10, num_outputs=2) == 2 * 5 + 2 * 10


def test_macs_avoided_is_dense_minus_event_driven():
    dense = 196_000
    edriven = 50_000
    out = macs_avoided(dense, edriven)
    assert out["macs_avoided"] == dense - edriven
    assert out["dense_macs"] == dense
    assert out["event_driven_macs"] == edriven
    assert out["macs_avoided_fraction"] == (dense - edriven) / dense


def test_profile_event_budget_uses_kept_events(mock_data):
    """At f=0.5, events_total in the profile must equal the count of retained events."""
    f = 0.5
    profile = profile_event_budget(
        mock_data["event_times"],
        mock_data["event_neurons"],
        fraction=f,
        num_neurons=int(mock_data["num_neurons"]),
    )
    expected_events = sum(
        max(1, int(np.floor(f * t.size))) if t.size else 0
        for t in mock_data["event_times"]
    )
    assert profile["events_total"] == expected_events


def test_compute_efficiency_summary_monotonic(mock_data):
    """MACs-avoided fraction must be monotonically non-decreasing as f shrinks."""
    summary = compute_efficiency_summary(mock_data, fractions=(1.0, 0.5, 0.25, 0.1))
    fractions = [row["macs_avoided_fraction"] for row in summary["budgets"]]
    for a, b in pairwise(fractions):
        assert b >= a, f"non-monotonic at {a} -> {b}"
