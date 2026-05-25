"""Linear readout from SNN hidden activity to 2D cursor velocity.

Owned by snn-latency-decoder.
"""
from __future__ import annotations
from sklearn.linear_model import Ridge

import numpy as np


class LinearReadout:
    """y_hat = W @ z + b, fit by ridge on (z_train, v_train)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LinearReadout":
        model = Ridge(self.alpha)
        model.fit(Z, y)
        self.W = model.coef_
        self.b = model.intercept_
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        W = self.W
        b = self.b
        y_hat = W @ Z + b
        return y_hat
        
