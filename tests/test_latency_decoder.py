"""Smoke tests for the pure-latency ridge decoder."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import velocity_r2
from src.models.latency_decoder import LatencyDecoder


def test_latency_decoder_runs_end_to_end(mock_data):
    decoder = LatencyDecoder(
        num_neurons=int(mock_data["num_neurons"]),
        bin_size_ms=int(mock_data["bin_size_ms"]),
        alpha=1.0,
        standardize=True,
    ).fit(
        mock_data["event_times"], mock_data["event_neurons"],
        mock_data["velocity"],
        mock_data["train_idx"], mock_data["val_idx"],
    )
    y_pred = decoder.predict(
        mock_data["event_times"], mock_data["event_neurons"], mock_data["test_idx"]
    )
    assert y_pred.shape == (mock_data["test_idx"].size, 2)
    r2 = velocity_r2(mock_data["velocity"][mock_data["test_idx"]], y_pred)
    assert np.isfinite(r2["r2_joint"])


def test_latency_decoder_alpha_sweep_picks_best(mock_data):
    decoder = LatencyDecoder(
        num_neurons=int(mock_data["num_neurons"]),
        bin_size_ms=int(mock_data["bin_size_ms"]),
    ).fit(
        mock_data["event_times"], mock_data["event_neurons"],
        mock_data["velocity"],
        mock_data["train_idx"], mock_data["val_idx"],
        alphas=(0.01, 1.0, 100.0),
    )
    assert decoder.best_alpha in (0.01, 1.0, 100.0)
    assert decoder.alpha_sweep is not None
    assert len(decoder.alpha_sweep) == 3
