"""Smoke test for the trainable BPTT SNN.

Keeps epochs and hidden size tiny so the test runs fast — just verifies
the forward/backward graph wires up and predictions land in the right
shape with a finite R^2.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import velocity_r2
from src.models.trained_snn import TrainedLatencySNN, _sparse_events_to_subbin_counts


def test_subbin_counts_match_event_counts(mock_data):
    """Total spikes binned across sub-bins must equal the original event count."""
    x = _sparse_events_to_subbin_counts(
        mock_data["event_times"], mock_data["event_neurons"],
        num_neurons=int(mock_data["num_neurons"]),
        bin_size_ms=int(mock_data["bin_size_ms"]),
        num_sub_bins=10,
    )
    for t in range(x.shape[0]):
        assert int(x[t].sum()) == int(mock_data["event_neurons"][t].size)


def test_trained_snn_runs_end_to_end(mock_data):
    snn = TrainedLatencySNN(
        num_neurons=int(mock_data["num_neurons"]),
        hidden_dim=8,
        tau_ms=8.0,
        threshold=0.3,
        bin_size_ms=int(mock_data["bin_size_ms"]),
        num_sub_bins=5,
        lr=1e-2,
        epochs=3,
        patience=3,
        seed=0,
    ).fit(
        mock_data["event_times"], mock_data["event_neurons"],
        mock_data["velocity"],
        mock_data["train_idx"], mock_data["val_idx"],
    )
    y_pred = snn.predict(
        mock_data["event_times"], mock_data["event_neurons"], mock_data["test_idx"]
    )
    assert y_pred.shape == (mock_data["test_idx"].size, 2)
    r2 = velocity_r2(mock_data["velocity"][mock_data["test_idx"]], y_pred)
    assert np.isfinite(r2["r2_joint"])
    assert len(snn.history) >= 1


def test_trained_snn_history_records_metrics(mock_data):
    snn = TrainedLatencySNN(
        num_neurons=int(mock_data["num_neurons"]),
        hidden_dim=8, tau_ms=8.0, threshold=0.3,
        bin_size_ms=int(mock_data["bin_size_ms"]),
        num_sub_bins=4, epochs=2, patience=2, seed=0,
    ).fit(
        mock_data["event_times"], mock_data["event_neurons"],
        mock_data["velocity"],
        mock_data["train_idx"], mock_data["val_idx"],
    )
    keys = {"epoch", "train_mse", "val_r2"}
    for row in snn.history:
        assert keys.issubset(row.keys())
