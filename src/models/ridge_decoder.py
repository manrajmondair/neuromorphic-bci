"""Ridge regression spike-count decoder with optional alpha sweep on val set.

Owned by data-ridge-baseline. The baseline learns a single linear map
W ∈ R^(num_neurons x 2) plus bias b that predicts 2D cursor velocity from
the (possibly budget-restricted) spike-count vector at each bin.

L2 strength is selected by sweeping a grid of alphas, training each on
the train split, scoring each on the validation split, and refitting the
chosen alpha on the train split. The held-out test split is never seen
during alpha selection.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# Wide log-spaced grid — covers the typical "alpha that wins" for ridge
# decoders on NLB-style motor cortex regressions.
DEFAULT_ALPHAS: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


class RidgeDecoder:
    """L2-regularized linear decoder: y_hat = X @ W + b."""

    def __init__(
        self,
        alpha: float | None = None,
        alphas: Sequence[float] | None = None,
    ) -> None:
        if alpha is None and alphas is None:
            alpha = 1.0
        if alpha is not None and alphas is not None:
            raise ValueError("pass `alpha` (single) or `alphas` (sweep), not both")
        self.alpha: float | None = alpha
        self.alphas: tuple[float, ...] | None = tuple(alphas) if alphas is not None else None
        self.model: Ridge | None = None
        self.best_alpha: float | None = None
        self.alpha_sweep: list[tuple[float, float]] | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RidgeDecoder":
        """Fit the decoder, optionally selecting alpha on a validation set.

        If `alphas` was provided to `__init__`, X_val and y_val are required
        and the alpha with highest joint R² on val is chosen, then refit on
        the training data alone.
        """
        if X_train.ndim != 2 or y_train.ndim != 2 or y_train.shape[1] != 2:
            raise ValueError(
                f"expected X_train [N, F] and y_train [N, 2]; got {X_train.shape}, {y_train.shape}"
            )

        if self.alphas is not None:
            if X_val is None or y_val is None:
                raise ValueError("alpha sweep requires X_val and y_val")
            self.alpha_sweep = []
            best_alpha: float | None = None
            best_score = -np.inf
            for a in self.alphas:
                m = Ridge(alpha=a).fit(X_train, y_train)
                y_pred_val = m.predict(X_val)
                score = _joint_r2(y_val, y_pred_val)
                self.alpha_sweep.append((float(a), float(score)))
                logger.info("ridge sweep alpha=%-10.4g  val r2_joint=%+.4f", a, score)
                if score > best_score:
                    best_score = score
                    best_alpha = float(a)
            self.best_alpha = best_alpha
            logger.info("ridge: best alpha=%g  val r2_joint=%+.4f", best_alpha, best_score)
            self.model = Ridge(alpha=best_alpha).fit(X_train, y_train)
        else:
            self.best_alpha = float(self.alpha)  # type: ignore[arg-type]
            self.model = Ridge(alpha=self.alpha).fit(X_train, y_train)
            logger.info("ridge: fit with fixed alpha=%g", self.alpha)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RidgeDecoder.predict() called before fit()")
        return self.model.predict(X)


def _joint_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Joint velocity R² used for alpha selection (mirrors evaluation.metrics)."""
    from src.evaluation.metrics import velocity_r2

    return velocity_r2(y_true, y_pred)["r2_joint"]
