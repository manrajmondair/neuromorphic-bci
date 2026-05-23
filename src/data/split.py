"""Train/val/test split utilities. Time-contiguous to avoid leakage.

The optional `boundary_gap` parameter drops bins on each side of every split
boundary. With `boundary_gap=1`, a 3-point central-difference velocity at any
training bin cannot depend on positions from val or test (and vice versa) —
this is what we use to guarantee zero label leakage across splits.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def time_contiguous_split(
    num_bins: int,
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
    boundary_gap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) as chronological contiguous blocks.

    With boundary_gap=g, the last g bins of each block and the first g of
    the next are dropped from any split. Those gap bins exist in the
    processed object's spike/velocity arrays but appear in no index array,
    so no model trains or evaluates on them — eliminating finite-difference
    velocity leakage across split boundaries.
    """
    assert abs(sum(fracs) - 1.0) < 1e-6, f"split fracs must sum to 1.0, got {sum(fracs)}"
    assert boundary_gap >= 0, f"boundary_gap must be >= 0, got {boundary_gap}"
    assert num_bins > 0, "num_bins must be positive"

    n_train_target = int(fracs[0] * num_bins)
    n_val_target = int(fracs[1] * num_bins)
    g = boundary_gap

    train_end = n_train_target - g
    val_start = n_train_target + g
    val_end = n_train_target + n_val_target - g
    test_start = n_train_target + n_val_target + g

    if train_end <= 0 or val_start >= val_end or test_start >= num_bins:
        raise ValueError(
            f"boundary_gap={g} too large for num_bins={num_bins} with fracs={fracs}; "
            f"resulting blocks would be empty"
        )

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(val_start, val_end, dtype=np.int64)
    test_idx = np.arange(test_start, num_bins, dtype=np.int64)

    logger.info(
        "time_contiguous_split: num_bins=%d fracs=%s gap=%d -> train=%d val=%d test=%d (dropped=%d)",
        num_bins,
        fracs,
        g,
        train_idx.size,
        val_idx.size,
        test_idx.size,
        num_bins - (train_idx.size + val_idx.size + test_idx.size),
    )
    return train_idx, val_idx, test_idx
