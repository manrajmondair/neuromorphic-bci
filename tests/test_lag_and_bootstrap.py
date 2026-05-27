"""Smoke tests for the lag-feature stack and bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.spike_counts import stack_lag_features


def test_lag_zero_returns_input_float32(mock_data):
    counts = mock_data["spike_counts"]
    out = stack_lag_features(counts, num_lags=0)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, counts.astype(np.float32))


def test_lag_features_have_right_shape(mock_data):
    counts = mock_data["spike_counts"]
    out = stack_lag_features(counts, num_lags=3)
    assert out.shape == (counts.shape[0], counts.shape[1] * 4)


def test_lag_features_carry_history_within_block(mock_data):
    """Bin t's k-th lag block must equal counts[t-k] inside a contiguous block."""
    counts = mock_data["spike_counts"]
    N = counts.shape[1]
    out = stack_lag_features(counts, num_lags=2)
    for t in range(2, counts.shape[0]):
        np.testing.assert_array_equal(out[t, 0:N], counts[t])
        np.testing.assert_array_equal(out[t, N : 2 * N], counts[t - 1])
        np.testing.assert_array_equal(out[t, 2 * N : 3 * N], counts[t - 2])


def test_lag_features_zero_pad_across_split_boundary(mock_data):
    """A row at the start of a split must have zeros in any lag column reaching back."""
    counts = mock_data["spike_counts"]
    N = counts.shape[1]
    out = stack_lag_features(counts, num_lags=3, split_starts=(0, 100, 200))
    # Bin 100 is the start of a new split: its lag-1/2/3 blocks should be zero
    # (otherwise they would reach across into the train block).
    assert np.all(out[100, N:] == 0.0)
    # Bin 101: lag-1 block is OK (counts[100]); lag-2 and lag-3 are zeros.
    np.testing.assert_array_equal(out[101, N : 2 * N], counts[100])
    assert np.all(out[101, 2 * N :] == 0.0)


def test_bootstrap_point_estimate_matches_velocity_r2():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(300, 2))
    y_hat = y + 0.2 * rng.normal(size=y.shape)
    point = velocity_r2(y, y_hat)
    boot = velocity_r2_bootstrap(y, y_hat, n_boot=200, seed=0)
    for key in ("r2_vx", "r2_vy", "r2_joint"):
        assert boot[key] == pytest.approx(point[key])
        assert boot[f"{key}_ci_lo"] <= point[key] + 1e-9
        assert boot[f"{key}_ci_hi"] >= point[key] - 1e-9


def test_bootstrap_invalid_ci_raises():
    y = np.zeros((10, 2))
    with pytest.raises(ValueError):
        velocity_r2_bootstrap(y, y, n_boot=10, ci=1.5)
