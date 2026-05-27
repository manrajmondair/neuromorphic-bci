"""Pure spike-latency ridge decoder.

For each bin, the feature vector is the per-neuron time-to-first-spike
in milliseconds (with neurons that never fired in the bin set to the
sentinel `bin_size_ms`). A linear ridge readout maps this latency
vector to 2D cursor velocity.

This decoder isolates the *latency* signal: it has no LIF dynamics and
no spike counts. Comparing its R² against the spike-count ridge decoder
answers "is there decoding information in the first-spike time that
isn't in the count?" Comparing against the SNN and shuffle decoders
isolates the contribution of dynamic LIF integration on top of the raw
latency feature.

Optionally subtracts the train-set mean and divides by std per neuron
to give the readout well-conditioned inputs (default on).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.features.latency_order import time_to_first_spike


class LatencyDecoder:
    """time_to_first_spike + ridge → 2D velocity."""

    def __init__(
        self,
        num_neurons: int,
        bin_size_ms: int,
        alpha: float = 1.0,
        standardize: bool = True,
    ):
        self.num_neurons = int(num_neurons)
        self.bin_size_ms = int(bin_size_ms)
        self.alpha = float(alpha)
        self.standardize = bool(standardize)
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.model: Ridge | None = None
        self.best_alpha: float | None = None
        self.alpha_sweep: list[tuple[float, float]] | None = None

    def _features(
        self, event_times: list[np.ndarray], event_neurons: list[np.ndarray]
    ) -> np.ndarray:
        return time_to_first_spike(
            event_times, event_neurons, self.num_neurons, self.bin_size_ms
        )

    def fit(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        velocity: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray | None = None,
        alphas: tuple[float, ...] | None = None,
    ) -> "LatencyDecoder":
        X = self._features(event_times, event_neurons)
        Xtr = X[train_idx]
        ytr = velocity[train_idx]

        if self.standardize:
            self.mu = Xtr.mean(axis=0)
            self.sigma = Xtr.std(axis=0) + 1e-6
            Xtr = (Xtr - self.mu) / self.sigma

        if alphas is not None and val_idx is not None:
            Xval_raw = X[val_idx]
            if self.standardize:
                Xval = (Xval_raw - self.mu) / self.sigma
            else:
                Xval = Xval_raw
            yval = velocity[val_idx]
            self.alpha_sweep = []
            best_alpha = float(alphas[0])
            best_score = -np.inf
            for a in alphas:
                m = Ridge(alpha=float(a)).fit(Xtr, ytr)
                from src.evaluation.metrics import velocity_r2
                score = velocity_r2(yval, m.predict(Xval))["r2_joint"]
                self.alpha_sweep.append((float(a), float(score)))
                if score > best_score:
                    best_score = score
                    best_alpha = float(a)
            self.best_alpha = best_alpha
            self.model = Ridge(alpha=best_alpha).fit(Xtr, ytr)
        else:
            self.best_alpha = float(self.alpha)
            self.model = Ridge(alpha=self.alpha).fit(Xtr, ytr)
        return self

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LatencyDecoder.predict() called before fit()")
        X = self._features(event_times, event_neurons)[idx]
        if self.standardize:
            X = (X - self.mu) / self.sigma
        return self.model.predict(X)
