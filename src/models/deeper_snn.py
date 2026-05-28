"""Deeper trained SNN variants — multi-layer LIF + optional recurrence.

Same supervised target (2D velocity) and same data interface as
`TrainedLatencySNN`, but with additional architectural flexibility:

  * `n_layers`           number of stacked LIF layers.
  * `recurrent`          if True, each hidden layer has a recurrent
                         weight that feeds the previous-timestep spike
                         pattern back into itself.

The goal is to quantify whether the gap between the trained SNN and
strong recurrent / transformer-class decoders is fundamentally a
*capacity* gap or a *context-window* gap. If a multi-layer recurrent
LIF still does not exceed lag-4 ridge at f=1.0 by much, then the
remaining headroom is in feature engineering, not model class.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

from src.models.trained_snn import (
    _SpikeFn,
    _sparse_events_to_subbin_counts,
    _stack_history,
)

logger = logging.getLogger(__name__)


class DeeperTrainedSNN:
    """Stacked LIF layers with optional within-layer recurrence."""

    def __init__(
        self,
        num_neurons: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        tau_ms: float = 10.0,
        threshold: float = 0.30,
        bin_size_ms: int = 50,
        num_sub_bins: int = 10,
        k_history: int = 4,
        recurrent: bool = False,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 15,
        surrogate_slope: float = 25.0,
        seed: int = 0,
    ):
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one layer")
        self.num_neurons = int(num_neurons)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.bin_size_ms = int(bin_size_ms)
        self.num_sub_bins = int(num_sub_bins)
        self.k_history = int(k_history)
        self.recurrent = bool(recurrent)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.surrogate_slope = float(surrogate_slope)
        self.seed = int(seed)

        # Layer weights — input projection + (optional) recurrence per layer.
        self._params: dict[str, torch.Tensor] = {}
        self.history: list[dict[str, float]] = []
        self.best_val_r2: float = float("-inf")

    @staticmethod
    def _joint_r2(y_true, y_pred):
        mean_v = y_true.mean(dim=0, keepdim=True)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - mean_v) ** 2).sum()
        return 1.0 - ss_res / (ss_tot + 1e-12)

    def _encode(self, x, W_list, R_list):
        """Stack of LIF layers; last-layer per-bin spike count -> [B, hidden_last]."""
        B, S, N = x.shape
        decay = float(np.exp(-(self.bin_size_ms / self.num_sub_bins) / self.tau_ms))
        # Layer 0 input: project sub-bin counts.
        injections = torch.einsum("bsn,hn->bsh", x, W_list[0])  # [B, S, H0]
        u = torch.zeros(B, W_list[0].shape[0], dtype=x.dtype, device=x.device)
        s_prev = torch.zeros(B, W_list[0].shape[0], dtype=x.dtype, device=x.device)
        z_last = torch.zeros(B, self.hidden_dims[-1], dtype=x.dtype, device=x.device)
        # We process layer 0 timestep-by-timestep, feeding s into layer 1+ at the end of each tick.
        for t in range(S):
            # Layer 0 step
            inj = injections[:, t, :]
            if self.recurrent and R_list[0] is not None:
                inj = inj + s_prev @ R_list[0].T
            u = u * decay + inj
            s = _SpikeFn.apply(u, self.threshold, self.surrogate_slope)
            u = u - s * self.threshold
            s_prev = s

            # Higher layers: collapse the per-timestep update through W_list[k]
            layer_in = s
            for k in range(1, len(W_list)):
                inj_k = layer_in @ W_list[k].T
                u_k = (
                    torch.zeros(B, W_list[k].shape[0], dtype=x.dtype, device=x.device)
                    if t == 0
                    else self._higher_state[k - 1]
                )
                u_k = u_k * decay + inj_k
                s_k = _SpikeFn.apply(u_k, self.threshold, self.surrogate_slope)
                u_k = u_k - s_k * self.threshold
                if not hasattr(self, "_higher_state") or len(self._higher_state) < k:
                    if t == 0:
                        self._higher_state = []
                if t == 0 and len(self._higher_state) <= k - 1:
                    self._higher_state.append(u_k)
                else:
                    self._higher_state[k - 1] = u_k
                if k == len(W_list) - 1:
                    z_last = z_last + s_k
                layer_in = s_k
        return z_last

    def fit(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        velocity: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> "DeeperTrainedSNN":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("deeper_snn: device=%s", device)

        x_all = _sparse_events_to_subbin_counts(
            event_times, event_neurons, self.num_neurons,
            self.bin_size_ms, self.num_sub_bins,
        )
        split_starts = (int(train_idx.min()), int(val_idx.min()))
        x_all = _stack_history(x_all, self.k_history, split_starts)

        x_train = torch.from_numpy(x_all[train_idx]).to(device)
        x_val = torch.from_numpy(x_all[val_idx]).to(device)
        y_train = torch.from_numpy(np.asarray(velocity[train_idx], dtype=np.float32)).to(device)
        y_val = torch.from_numpy(np.asarray(velocity[val_idx], dtype=np.float32)).to(device)

        # Allocate params: W_list, optional R_list, readout.
        scale0 = 1.0 / np.sqrt(self.num_neurons)
        W_list_params: list[nn.Parameter] = [
            nn.Parameter(torch.randn(self.hidden_dims[0], self.num_neurons, device=device) * scale0)
        ]
        for k in range(1, len(self.hidden_dims)):
            scale_k = 1.0 / np.sqrt(self.hidden_dims[k - 1])
            W_list_params.append(
                nn.Parameter(
                    torch.randn(self.hidden_dims[k], self.hidden_dims[k - 1], device=device) * scale_k
                )
            )
        R_list_params: list[nn.Parameter | None] = []
        if self.recurrent:
            for k in range(len(self.hidden_dims)):
                scale_r = 1.0 / np.sqrt(self.hidden_dims[k])
                R_list_params.append(
                    nn.Parameter(
                        torch.randn(self.hidden_dims[k], self.hidden_dims[k], device=device)
                        * scale_r * 0.3
                    )
                )
        else:
            R_list_params = [None] * len(self.hidden_dims)

        W_out = nn.Parameter(
            torch.randn(2, self.hidden_dims[-1], device=device) * (1.0 / np.sqrt(self.hidden_dims[-1]))
        )
        b_out = nn.Parameter(torch.zeros(2, device=device))
        self._device = device

        params = list(W_list_params) + [p for p in R_list_params if p is not None] + [W_out, b_out]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        best_val_r2 = float("-inf")
        best = {
            "W_list": [w.detach().clone() for w in W_list_params],
            "R_list": [r.detach().clone() if r is not None else None for r in R_list_params],
            "W_out": W_out.detach().clone(),
            "b_out": b_out.detach().clone(),
        }
        bad = 0
        for epoch in range(self.epochs):
            self._higher_state = []  # reset transient state for this fwd pass
            z = self._encode(x_train, W_list_params, R_list_params)
            y_pred = z @ W_out.T + b_out
            loss = torch.mean((y_pred - y_train) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                self._higher_state = []
                z_val = self._encode(x_val, W_list_params, R_list_params)
                y_pred_val = z_val @ W_out.T + b_out
                val_r2 = float(self._joint_r2(y_val, y_pred_val).item())
            self.history.append({"epoch": epoch, "train_mse": float(loss.item()), "val_r2": val_r2})
            if val_r2 > best_val_r2 + 1e-5:
                best_val_r2 = val_r2
                best = {
                    "W_list": [w.detach().clone() for w in W_list_params],
                    "R_list": [r.detach().clone() if r is not None else None for r in R_list_params],
                    "W_out": W_out.detach().clone(),
                    "b_out": b_out.detach().clone(),
                }
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self._params = best
        self.best_val_r2 = float(best_val_r2)
        return self

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
        split_starts: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("DeeperTrainedSNN.predict() called before fit()")
        x_all = _sparse_events_to_subbin_counts(
            event_times, event_neurons, self.num_neurons,
            self.bin_size_ms, self.num_sub_bins,
        )
        if split_starts is None:
            split_starts = (int(np.min(idx)),)
        x_all = _stack_history(x_all, self.k_history, split_starts)
        device = getattr(self, "_device", self._params["W_out"].device)
        x = torch.from_numpy(x_all[idx]).to(device)
        self._higher_state = []
        with torch.no_grad():
            z = self._encode(x, self._params["W_list"], self._params["R_list"])
            y_pred = z @ self._params["W_out"].T + self._params["b_out"]
        return y_pred.cpu().numpy()
