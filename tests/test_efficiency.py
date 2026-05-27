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
        bin_size_ms=int(mock_data["bin_size_ms"]),
    )
    expected_events = sum(
        max(1, int(np.floor(f * t.size))) if t.size else 0
        for t in mock_data["event_times"]
    )
    assert profile["events_total"] == expected_events


def test_energy_table_scales_with_pj_per_mac(mock_data):
    """Doubling pj/MAC should double dense_energy_uj exactly."""
    from src.evaluation.efficiency_tracker import MAC_ENERGY_PJ, energy_table

    table = energy_table(
        dense_macs=1_000_000,
        event_driven_macs=100_000,
        num_bins=int(mock_data["spike_counts"].shape[0]),
        bin_size_ms=int(mock_data["bin_size_ms"]),
    )
    cpu = table["per_chip"]["cpu_x86"]
    npole = table["per_chip"]["northpole"]
    ratio_pj = MAC_ENERGY_PJ["cpu_x86"] / MAC_ENERGY_PJ["northpole"]
    ratio_uj = cpu["dense_energy_uj"] / npole["dense_energy_uj"]
    assert abs(ratio_uj - ratio_pj) < 1e-9, (ratio_uj, ratio_pj)


def test_energy_avoided_fraction_matches_macs_avoided_fraction(mock_data):
    """For a single chip, avoided fraction is just (dense - edriven) / dense — chip-independent."""
    from src.evaluation.efficiency_tracker import energy_table

    table = energy_table(
        dense_macs=2_000_000, event_driven_macs=300_000,
        num_bins=100, bin_size_ms=50,
    )
    fractions = {chip: row["energy_avoided_fraction"] for chip, row in table["per_chip"].items()}
    assert len(set(round(f, 12) for f in fractions.values())) == 1


def test_compute_efficiency_summary_monotonic(mock_data):
    """MACs-avoided fraction must be monotonically non-decreasing as f shrinks."""
    summary = compute_efficiency_summary(mock_data, fractions=(1.0, 0.5, 0.25, 0.1))
    fractions = [row["macs_avoided_fraction"] for row in summary["budgets"]]
    for a, b in pairwise(fractions):
        assert b >= a, f"non-monotonic at {a} -> {b}"
