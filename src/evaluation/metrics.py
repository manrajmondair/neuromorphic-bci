"""Shared decoding-quality metrics.

The primary metric is the joint velocity R² from the project proposal:

    R² = 1 - sum_t ||v_t - v_hat_t||²  /  sum_t ||v_t - mean(v)||²

where the squared norm is taken over the 2D velocity vector at each bin.
We also report the per-axis R² (`r2_vx`, `r2_vy`) for interpretability,
but `r2_joint` is the single number the headline figure ranks models on.

Every model in this repo must score itself through `velocity_r2` so the
ridge baseline, SNN, and order-shuffle control are strictly comparable.
"""
from __future__ import annotations

import numpy as np


def velocity_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Joint and per-axis R² for 2D velocity predictions.

    `r2_joint` matches the proposal's formula exactly: sums of squared
    L2 residuals over both axes in the numerator and sums of squared
    L2 deviations from the empirical mean velocity in the denominator.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError(f"y_true / y_pred must be [T, 2], got {y_true.shape}")
    if y_true.shape[0] < 2:
        raise ValueError("need at least 2 samples to compute R²")

    mean_v = y_true.mean(axis=0, keepdims=True)
    sq_res = (y_true - y_pred) ** 2
    sq_tot = (y_true - mean_v) ** 2

    ss_res_axis = sq_res.sum(axis=0)  # [2]
    ss_tot_axis = sq_tot.sum(axis=0)  # [2]
    r2_axis = 1.0 - ss_res_axis / np.where(ss_tot_axis == 0, 1.0, ss_tot_axis)

    ss_res_joint = ss_res_axis.sum()
    ss_tot_joint = ss_tot_axis.sum()
    if ss_tot_joint == 0:
        r2_joint = float("nan")
    else:
        r2_joint = 1.0 - ss_res_joint / ss_tot_joint

    return {
        "r2_vx": float(r2_axis[0]),
        "r2_vy": float(r2_axis[1]),
        "r2_joint": float(r2_joint),
    }
