"""Joint velocity R² metric — the single number every model in the project
is ranked on, so it gets its own focused tests against analytical cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import velocity_r2


def test_perfect_prediction_gives_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(100, 2)).astype(np.float32)
    r2 = velocity_r2(y, y)
    assert r2["r2_joint"] == pytest.approx(1.0)
    assert r2["r2_vx"] == pytest.approx(1.0)
    assert r2["r2_vy"] == pytest.approx(1.0)


def test_mean_prediction_gives_zero():
    """Predicting the empirical mean yields R² = 0 by construction."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=(500, 2)).astype(np.float32)
    y_pred = np.broadcast_to(y.mean(axis=0), y.shape)
    r2 = velocity_r2(y, y_pred)
    assert r2["r2_joint"] == pytest.approx(0.0, abs=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        velocity_r2(np.zeros((10, 2)), np.zeros((9, 2)))


def test_one_dim_input_raises():
    with pytest.raises(ValueError):
        velocity_r2(np.zeros((10,)), np.zeros((10,)))


def test_joint_matches_proposal_formula():
    """The joint R² in metrics.py must match the explicit
    1 - sum||v_t - v_hat_t||^2 / sum||v_t - mean(v)||^2 formula from proposal §4.1.
    """
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=(200, 2)).astype(np.float64)
    y_pred = y_true + 0.3 * rng.normal(size=(200, 2))

    mean_v = y_true.mean(axis=0, keepdims=True)
    ss_res_expected = np.sum((y_true - y_pred) ** 2)
    ss_tot_expected = np.sum((y_true - mean_v) ** 2)
    expected = 1.0 - ss_res_expected / ss_tot_expected

    r2 = velocity_r2(y_true, y_pred)
    assert r2["r2_joint"] == pytest.approx(expected, rel=1e-10)
