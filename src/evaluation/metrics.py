"""Shared metric functions. Every model must call these — not its own R²."""
from __future__ import annotations

import numpy as np


def velocity_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Per-axis and mean R² for 2D velocity predictions.

    R²_d = 1 - SS_res_d / SS_tot_d, computed per axis against the mean of
    y_true on that axis. Returns {"r2_vx", "r2_vy", "r2_mean"}.
    """
    assert y_true.shape == y_pred.shape and y_true.shape[1] == 2
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)
    return {
        "r2_vx": float(r2[0]),
        "r2_vy": float(r2[1]),
        "r2_mean": float(r2.mean()),
    }
