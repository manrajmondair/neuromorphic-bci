"""Linear readout from SNN hidden activity to 2D cursor velocity.

Owned by snn-latency-decoder.
"""
from __future__ import annotations

import numpy as np


class LinearReadout:
    """y_hat = W @ z + b, fit by ridge on (z_train, v_train)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LinearReadout":
        raise NotImplementedError

    def predict(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
