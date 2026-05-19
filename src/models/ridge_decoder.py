"""Ridge regression spike-count decoder. Owned by data-ridge-baseline."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class RidgeDecoder:
    """Thin wrapper around sklearn.Ridge for [num_bins, num_neurons] → [num_bins, 2]."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeDecoder":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
