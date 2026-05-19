"""Train/val/test split utilities. Time-contiguous to avoid leakage."""
from __future__ import annotations

import numpy as np


def time_contiguous_split(
    num_bins: int,
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) as chronological contiguous blocks."""
    assert abs(sum(fracs) - 1.0) < 1e-6, "split fracs must sum to 1.0"
    n_train = int(fracs[0] * num_bins)
    n_val = int(fracs[1] * num_bins)
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, num_bins, dtype=np.int64)
    return train_idx, val_idx, test_idx
