"""Linear readout from SNN hidden activity to 2D cursor velocity.

Owned by snn-latency-decoder.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class LinearReadout:
    """y_hat = Z @ W^T + b, fit by ridge on (z_train, v_train)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Ridge | None = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> LinearReadout:
        if Z.ndim != 2:
            raise ValueError(f"Z must be [N, hidden], got {Z.shape}")
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"y must be [N, 2], got {y.shape}")
        if Z.shape[0] != y.shape[0]:
            raise ValueError(f"Z and y must agree on N: {Z.shape[0]} vs {y.shape[0]}")
        self.model = Ridge(alpha=self.alpha).fit(Z, y)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LinearReadout.predict() called before fit()")
        return self.model.predict(Z)
