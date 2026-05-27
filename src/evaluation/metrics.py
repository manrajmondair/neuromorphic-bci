"""Shared decoding-quality metrics.

The primary metric is the joint velocity R² from the project proposal:

    R² = 1 - sum_t ||v_t - v_hat_t||²  /  sum_t ||v_t - mean(v)||²

where the squared norm is taken over the 2D velocity vector at each bin.
We also report the per-axis R² (`r2_vx`, `r2_vy`) for interpretability,
but `r2_joint` is the single number the headline figure ranks models on.

`velocity_r2_bootstrap` resamples the test bins with replacement (default
1000 reps) and returns 2.5/97.5 percentile bounds for each R². This is
the CI convention every result row in the canonical JSONs reports.

Every model in this repo must score itself through `velocity_r2` so the
ridge baseline, SNN, and order-shuffle control are strictly comparable.
"""
from __future__ import annotations

import numpy as np


def velocity_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Joint and per-axis R² for 2D velocity predictions."""
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


def velocity_r2_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """Bootstrap 95% confidence intervals for each velocity R².

    Resamples test bins with replacement `n_boot` times. Returns the
    point estimate plus low/high CI bounds for `r2_vx`, `r2_vy`,
    `r2_joint`. The point estimate matches `velocity_r2` on the full
    sample; CI bounds are the 2.5 and 97.5 percentiles across resamples
    (for the default ci=0.95).
    """
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    point = velocity_r2(y_true, y_pred)

    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    samples = {"r2_vx": np.empty(n_boot), "r2_vy": np.empty(n_boot), "r2_joint": np.empty(n_boot)}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = velocity_r2(y_true[idx], y_pred[idx])
        samples["r2_vx"][b] = r["r2_vx"]
        samples["r2_vy"][b] = r["r2_vy"]
        samples["r2_joint"][b] = r["r2_joint"]

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    out: dict[str, float] = {}
    for key in ("r2_vx", "r2_vy", "r2_joint"):
        out[key] = float(point[key])
        out[f"{key}_ci_lo"] = float(np.percentile(samples[key], lo_pct))
        out[f"{key}_ci_hi"] = float(np.percentile(samples[key], hi_pct))
        out[f"{key}_boot_std"] = float(np.std(samples[key]))
    out["n_boot"] = int(n_boot)
    out["ci"] = float(ci)
    return out
