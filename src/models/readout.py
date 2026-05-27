"""Linear and small-MLP readouts from SNN hidden activity to 2D cursor velocity.

Three flavours live here so the SNN-readout ablation can swap them in
without touching the encoder:

  * `LinearReadout`         — closed-form ridge (sklearn)
  * `SGDLinearReadout`      — same linear model trained by SGD with
                              early stopping on val
  * `MLPReadout`            — small 2-layer MLP (ReLU) trained by Adam
                              with early stopping on val

All three implement `fit(Z, y, Z_val=None, y_val=None) -> self` and
`predict(Z) -> np.ndarray[N, 2]`.

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

    def fit(self, Z, y, Z_val=None, y_val=None) -> "LinearReadout":  # noqa: ARG002
        Z = np.asarray(Z, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if Z.ndim != 2:
            raise ValueError(f"Z must be [N, hidden], got {Z.shape}")
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"y must be [N, 2], got {y.shape}")
        if Z.shape[0] != y.shape[0]:
            raise ValueError(f"Z and y must agree on N: {Z.shape[0]} vs {y.shape[0]}")
        self.model = Ridge(alpha=self.alpha).fit(Z, y)
        return self

    def predict(self, Z):
        if self.model is None:
            raise RuntimeError("LinearReadout.predict() called before fit()")
        return self.model.predict(np.asarray(Z, dtype=np.float32))


class SGDLinearReadout:
    """Linear y_hat = Z @ W^T + b trained by Adam with early stopping on val.

    Provides a non-closed-form linear comparison point: same hypothesis
    class as `LinearReadout` but with optimization noise. If this matches
    LinearReadout, the readout is at the closed-form optimum; if not,
    the ridge problem is poorly conditioned or the optimizer is.
    """

    def __init__(
        self,
        weight_decay: float = 1e-3,
        lr: float = 1e-2,
        max_epochs: int = 200,
        patience: int = 10,
        batch_size: int = 256,
        seed: int = 0,
    ):
        self.weight_decay = float(weight_decay)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, Z, y, Z_val=None, y_val=None) -> "SGDLinearReadout":
        import torch

        torch.manual_seed(self.seed)
        Z_t = torch.from_numpy(np.asarray(Z, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        N, D = Z_t.shape
        W = torch.zeros(2, D, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=self.lr, weight_decay=self.weight_decay)

        if Z_val is not None and y_val is not None:
            Zv = torch.from_numpy(np.asarray(Z_val, dtype=np.float32))
            yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
        else:
            Zv = yv = None

        best_val = float("inf")
        best_W = W.detach().clone()
        best_b = b.detach().clone()
        bad = 0
        for _epoch in range(self.max_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                pred = Z_t[idx] @ W.T + b
                loss = torch.mean((pred - y_t[idx]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if Zv is not None:
                with torch.no_grad():
                    val_loss = torch.mean((Zv @ W.T + b - yv) ** 2).item()
                if val_loss < best_val - 1e-7:
                    best_val = val_loss
                    best_W = W.detach().clone()
                    best_b = b.detach().clone()
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                best_W = W.detach().clone()
                best_b = b.detach().clone()

        self.W = best_W.numpy()
        self.b = best_b.numpy()
        return self

    def predict(self, Z):
        if self.W is None or self.b is None:
            raise RuntimeError("SGDLinearReadout.predict() called before fit()")
        Z = np.asarray(Z, dtype=np.float32)
        return Z @ self.W.T + self.b


class MLPReadout:
    """Small 2-layer ReLU MLP readout: Z → ReLU(W1 Z + b1) → W2 . + b2 → [N, 2].

    Trained with Adam + weight decay + early stopping on val.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        weight_decay: float = 1e-3,
        lr: float = 1e-3,
        max_epochs: int = 300,
        patience: int = 15,
        batch_size: int = 256,
        seed: int = 0,
    ):
        self.hidden_dim = int(hidden_dim)
        self.weight_decay = float(weight_decay)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._model = None

    def fit(self, Z, y, Z_val=None, y_val=None) -> "MLPReadout":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        Z_t = torch.from_numpy(np.asarray(Z, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        N, D = Z_t.shape
        model = nn.Sequential(
            nn.Linear(D, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if Z_val is not None and y_val is not None:
            Zv = torch.from_numpy(np.asarray(Z_val, dtype=np.float32))
            yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
        else:
            Zv = yv = None

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bad = 0
        for _epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                pred = model(Z_t[idx])
                loss = torch.mean((pred - y_t[idx]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if Zv is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = torch.mean((model(Zv) - yv) ** 2).item()
                if val_loss < best_val - 1e-7:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        return self

    def predict(self, Z):
        import torch
        if self._model is None:
            raise RuntimeError("MLPReadout.predict() called before fit()")
        with torch.no_grad():
            out = self._model(torch.from_numpy(np.asarray(Z, dtype=np.float32)))
        return out.numpy()
