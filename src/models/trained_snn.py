"""End-to-end trainable LIF SNN with surrogate gradients.

Architecture
------------
Each 50 ms input bin is re-binned into a fixed number of equal-width
sub-bins (default 10, i.e. 5 ms each), producing a per-bin tensor of
input spike counts with shape `[num_sub_bins, num_neurons]`. We then
run a hidden LIF layer over the sub-bins with continuous-time leak,
fully vectorized across all bins:

    for t = 0..num_sub_bins - 1:
        u = u * exp(-sub_bin_ms / tau_ms) + x[:, t, :] @ W^T
        s = surrogate_spike(u, threshold)        # fast-sigmoid backward
        z += s
        u = u - s * threshold                    # hard reset

The per-bin hidden spike-count vector `z` (of shape `[num_bins, hidden_dim]`)
goes through a linear readout to 2D velocity. All weights — input
projection `W`, readout `W_out` and `b_out` — are trained by Adam with
backprop-through-time using snntorch's fast-sigmoid surrogate gradient.

This is the trained counterpart to the random-projection reservoir in
`SparseLatencySNN`. If a trained SNN can match or beat the count-ridge
baseline at low event budgets while the reservoir SNN cannot, the
proposal's claim ("a sparse-event neuromorphic decoder is sufficient")
survives the test.

Sub-bin coarsening trades sub-`sub_bin_ms` ms timing precision for a
fully vectorized BPTT pass. For BCI cursor decoding at 50 ms bins the
default 5 ms sub-bin resolution is far below behaviourally relevant.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class _SpikeFn(torch.autograd.Function):
    """Heaviside in forward; fast-sigmoid surrogate in backward."""

    @staticmethod
    def forward(ctx, u, threshold, slope):
        ctx.save_for_backward(u)
        ctx.threshold = float(threshold)
        ctx.slope = float(slope)
        return (u >= threshold).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (u,) = ctx.saved_tensors
        x = ctx.slope * (u - ctx.threshold)
        surrogate_grad = ctx.slope / (1.0 + x.abs()) ** 2
        return grad_out * surrogate_grad, None, None


def _sparse_events_to_subbin_counts(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    num_neurons: int,
    bin_size_ms: int,
    num_sub_bins: int,
) -> np.ndarray:
    """Bucket each bin's sparse events into `num_sub_bins` equal-width slots.

    Returns a `[num_bins, num_sub_bins, num_neurons]` float32 array of
    spike counts. `event_times[t]` is in ms within the parent bin.
    """
    num_bins = len(event_times)
    sub_w = bin_size_ms / num_sub_bins
    out = np.zeros((num_bins, num_sub_bins, num_neurons), dtype=np.float32)
    for t in range(num_bins):
        times = event_times[t]
        neurons = event_neurons[t]
        if neurons.size == 0:
            continue
        slot = np.clip((times / sub_w).astype(np.int64), 0, num_sub_bins - 1)
        # Group-by accumulation.
        np.add.at(out[t], (slot, neurons.astype(np.int64)), 1.0)
    return out


class TrainedLatencySNN:
    """Trainable LIF + linear readout, supervised on 2D velocity."""

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int = 128,
        tau_ms: float = 10.0,
        threshold: float = 1.0,
        bin_size_ms: int = 50,
        num_sub_bins: int = 10,
        lr: float = 5e-3,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        patience: int = 10,
        batch_size: int = 0,  # 0 = full batch
        surrogate_slope: float = 25.0,
        seed: int = 0,
    ):
        if num_sub_bins < 1:
            raise ValueError(f"num_sub_bins must be >= 1, got {num_sub_bins}")
        self.num_neurons = int(num_neurons)
        self.hidden_dim = int(hidden_dim)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.bin_size_ms = int(bin_size_ms)
        self.num_sub_bins = int(num_sub_bins)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.surrogate_slope = float(surrogate_slope)
        self.seed = int(seed)

        self._W: torch.Tensor | None = None
        self._W_out: torch.Tensor | None = None
        self._b_out: torch.Tensor | None = None
        self.history: list[dict[str, float]] = []
        self.best_val_r2: float = float("-inf")

    @staticmethod
    def _joint_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        mean_v = y_true.mean(dim=0, keepdim=True)
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - mean_v) ** 2).sum()
        return 1.0 - ss_res / (ss_tot + 1e-12)

    def _encode(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """LIF forward over sub-bins.

        x : [B, S, N]   input spike counts (B bins, S sub-bins, N neurons)
        W : [hidden, N] input projection
        returns z : [B, hidden] per-bin total hidden spike counts.
        """
        B, S, N = x.shape
        H = W.shape[0]
        decay = float(np.exp(-(self.bin_size_ms / self.num_sub_bins) / self.tau_ms))
        u = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        # Pre-project all sub-bin inputs at once for efficiency.
        injections = torch.einsum("bsn,hn->bsh", x, W)  # [B, S, H]
        for t in range(S):
            u = u * decay + injections[:, t, :]
            s = _SpikeFn.apply(u, self.threshold, self.surrogate_slope)
            z = z + s
            u = u - s * self.threshold
        return z

    def fit(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        velocity: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> "TrainedLatencySNN":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Pre-bucket events once (no python loop in the inner training loop).
        x_all = _sparse_events_to_subbin_counts(
            event_times, event_neurons, self.num_neurons,
            self.bin_size_ms, self.num_sub_bins,
        )
        x_train = torch.from_numpy(x_all[train_idx])
        x_val = torch.from_numpy(x_all[val_idx])
        y_train = torch.from_numpy(np.asarray(velocity[train_idx], dtype=np.float32))
        y_val = torch.from_numpy(np.asarray(velocity[val_idx], dtype=np.float32))

        scale = 1.0 / np.sqrt(self.num_neurons)
        W = nn.Parameter(torch.randn(self.hidden_dim, self.num_neurons) * scale)
        W_out = nn.Parameter(torch.randn(2, self.hidden_dim) * (1.0 / np.sqrt(self.hidden_dim)))
        b_out = nn.Parameter(torch.zeros(2))
        opt = torch.optim.Adam([W, W_out, b_out], lr=self.lr, weight_decay=self.weight_decay)

        best_val_r2 = float("-inf")
        best_state = {"W": W.detach().clone(), "W_out": W_out.detach().clone(), "b_out": b_out.detach().clone()}
        bad = 0

        n_train = x_train.shape[0]
        batch = self.batch_size if self.batch_size > 0 else n_train

        for epoch in range(self.epochs):
            perm = rng.permutation(n_train) if self.batch_size > 0 else np.arange(n_train)
            train_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch):
                idx = perm[start : start + batch]
                xb = x_train[idx]
                yb = y_train[idx]
                z = self._encode(xb, W)
                y_pred = z @ W_out.T + b_out
                loss = torch.mean((y_pred - yb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += float(loss.item())
                n_batches += 1
            train_loss /= max(n_batches, 1)

            with torch.no_grad():
                z_val = self._encode(x_val, W)
                y_pred_val = z_val @ W_out.T + b_out
                val_r2 = float(self._joint_r2(y_val, y_pred_val).item())
            self.history.append({"epoch": epoch, "train_mse": float(train_loss), "val_r2": val_r2})

            if val_r2 > best_val_r2 + 1e-5:
                best_val_r2 = val_r2
                best_state = {
                    "W": W.detach().clone(),
                    "W_out": W_out.detach().clone(),
                    "b_out": b_out.detach().clone(),
                }
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self._W = best_state["W"]
        self._W_out = best_state["W_out"]
        self._b_out = best_state["b_out"]
        self.best_val_r2 = float(best_val_r2)
        return self

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
    ) -> np.ndarray:
        if self._W is None or self._W_out is None or self._b_out is None:
            raise RuntimeError("TrainedLatencySNN.predict() called before fit()")
        x_all = _sparse_events_to_subbin_counts(
            event_times, event_neurons, self.num_neurons,
            self.bin_size_ms, self.num_sub_bins,
        )
        x = torch.from_numpy(x_all[idx])
        with torch.no_grad():
            z = self._encode(x, self._W)
            y_pred = z @ self._W_out.T + self._b_out
        return y_pred.numpy()
