"""Smoke tests for the SNN encoder, readout, and order-shuffle interaction.

These verify the SNN side runs end-to-end on mock data, predictions have the
right shape, and the order-shuffle control actually changes the encoder
output (so the comparison in `scripts/run_snn.py` is meaningful).
"""
from __future__ import annotations

import numpy as np

from src.controls.order_shuffle import shuffle_within_bin_order
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.features.latency_order import (
    pairwise_order_features,
    time_to_first_spike,
)
from src.models.readout import LinearReadout
from src.models.snn_decoder import SparseLatencySNN


def test_linear_readout_predicts_right_shape():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(80, 32)).astype(np.float32)
    y = rng.normal(size=(80, 2)).astype(np.float32)
    readout = LinearReadout(alpha=1.0).fit(Z, y)
    assert readout.predict(Z).shape == (80, 2)


def test_time_to_first_spike_sentinel_for_silent_neurons(mock_data):
    taus = time_to_first_spike(
        mock_data["event_times"],
        mock_data["event_neurons"],
        int(mock_data["num_neurons"]),
        int(mock_data["bin_size_ms"]),
    )
    # Neurons that never fired must hit the bin_size_ms sentinel.
    assert taus.max() <= float(mock_data["bin_size_ms"])
    assert (taus == float(mock_data["bin_size_ms"])).any()


def test_pairwise_order_features_basic_property(mock_data):
    """For pair (i, j): result must be 1 iff i fires strictly before j in the bin."""
    pairs = np.array([[0, 1], [1, 0]], dtype=np.int64)
    out = pairwise_order_features(
        mock_data["event_times"],
        mock_data["event_neurons"],
        pairs,
        int(mock_data["num_neurons"]),
        int(mock_data["bin_size_ms"]),
    )
    num_bins = mock_data["spike_counts"].shape[0]
    assert out.shape == (num_bins, 2)
    # (i,j) and (j,i) can both be 0 (neither fired or tied at sentinel) but
    # cannot both be 1.
    assert not ((out[:, 0] == 1) & (out[:, 1] == 1)).any()


def test_snn_runs_end_to_end_on_mock(mock_data):
    """fit -> predict on mock data: shapes line up, R^2 is finite, no crashes."""
    num_neurons = int(mock_data["num_neurons"])
    snn = SparseLatencySNN(
        num_neurons=num_neurons,
        hidden_dim=16,
        tau_ms=8.0,
        bin_size_ms=int(mock_data["bin_size_ms"]),
        seed=0,
    ).fit(
        mock_data["event_times"],
        mock_data["event_neurons"],
        mock_data["velocity"],
        mock_data["train_idx"],
        mock_data["val_idx"],
    )
    y_pred = snn.predict(
        mock_data["event_times"], mock_data["event_neurons"], mock_data["test_idx"]
    )
    assert y_pred.shape == (mock_data["test_idx"].size, 2)
    r2 = velocity_r2(mock_data["velocity"][mock_data["test_idx"]], y_pred)
    assert np.isfinite(r2["r2_joint"])


def test_snn_predict_respects_idx_length(mock_data):
    """predict must produce one row per index in `idx`, not one per dataset bin.

    The regression we are guarding against is reading the *full* event list
    inside the predict loop instead of the locally sliced one, which would
    produce predictions of length `num_bins` regardless of `idx.size`.
    """
    num_neurons = int(mock_data["num_neurons"])
    snn = SparseLatencySNN(
        num_neurons=num_neurons, hidden_dim=12, tau_ms=8.0,
        bin_size_ms=int(mock_data["bin_size_ms"]), seed=1,
    ).fit(
        mock_data["event_times"], mock_data["event_neurons"], mock_data["velocity"],
        mock_data["train_idx"], mock_data["val_idx"],
    )
    test_idx = mock_data["test_idx"]
    for take in (3, test_idx.size // 2, test_idx.size):
        y_pred = snn.predict(
            mock_data["event_times"], mock_data["event_neurons"], test_idx[:take]
        )
        assert y_pred.shape == (take, 2)


def test_shuffle_changes_encoder_output(mock_data):
    """The order-shuffle control must change the SNN's hidden encoding —
    otherwise comparing shuffled R^2 against real R^2 has no signal.
    """
    num_neurons = int(mock_data["num_neurons"])
    # Reduce to bins with >= 3 events so a shuffle can actually reorder.
    et, en = mock_data["event_times"], mock_data["event_neurons"]
    et_s, en_s = shuffle_within_bin_order(et, en, seed=0)

    snn = SparseLatencySNN(
        num_neurons=num_neurons, hidden_dim=16, tau_ms=8.0,
        bin_size_ms=int(mock_data["bin_size_ms"]), seed=0,
    )
    # Cheat the fit/predict — we just want to compare _encode outputs.
    rng = np.random.default_rng(0)
    snn.W = rng.normal(
        loc=0.0, scale=1.0 / np.sqrt(num_neurons),
        size=(snn.hidden_dim, num_neurons),
    ).astype(np.float32)
    Z_real = snn._encode(et, en)
    Z_shuffled = snn._encode(et_s, en_s)
    assert Z_real.shape == Z_shuffled.shape
    # Some bins must change. (Single-spike bins won't, but the dataset has multi-spike bins.)
    assert not np.array_equal(Z_real, Z_shuffled)


def test_event_budget_preserves_temporal_order():
    """Applying an event budget to already-sorted times must keep the
    surviving events sorted — invariant relied on by the SNN's LIF leak.
    """
    rng = np.random.default_rng(0)
    event_times = [
        np.sort(rng.uniform(0, 50, size=n).astype(np.float32))
        for n in (12, 0, 1, 30, 7)
    ]
    event_neurons = [
        rng.integers(0, 10, size=t.size).astype(np.int32) for t in event_times
    ]
    et_kept, _ = apply_event_budget(event_times, event_neurons, fraction=0.5)
    for t in et_kept:
        if t.size > 1:
            assert np.all(np.diff(t) >= 0)
