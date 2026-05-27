"""End-to-end trainable LIF SNN with surrogate gradients, optional bin history.

Architecture
------------
Each input bin is re-binned into a fixed number of equal-width sub-bins
(default 10 sub-bins of 5 ms at the 50 ms bin width), producing a
per-bin tensor of input spike counts with shape `[num_sub_bins, num_neurons]`.

If `k_history > 0`, the encoder concatenates the previous `k_history`
bins of sub-bin counts along the time axis before running the LIF, so
the LIF sees `(k_history + 1) * num_sub_bins` time steps per prediction.
History columns that would reach across a train/val/test split boundary
are zero-padded (the same idea as `stack_lag_features` for the ridge
decoder).

The hidden LIF layer is run vectorized across bins:

    for t = 0..S - 1:
        u = u * exp(-sub_bin_ms / tau_ms) + x[:, t, :] @ W^T
        s = surrogate_spike(u, threshold)        # fast-sigmoid backward
        z += s                                   # per-bin spike count
        u = u - s * threshold                    # hard reset

`z` (shape `[num_bins, hidden_dim]`) is the feature vector passed to a
linear readout to predict 2D velocity. All weights — input projection
`W`, readout `W_out` and `b_out` — are trained by Adam with BPTT and a
fast-sigmoid surrogate gradient.

History lets the LIF integrate over the same temporal context the
lag-feature ridge decoder uses, which is what closes the accuracy gap
to the strong baseline.
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

    Returns `[num_bins, num_sub_bins, num_neurons]` float32.
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
        np.add.at(out[t], (slot, neurons.astype(np.int64)), 1.0)
    return out


def _stack_history(
    x: np.ndarray,
    k_history: int,
    split_starts: tuple[int, ...] | None,
) -> np.ndarray:
    """Stack `k_history` previous bins along the sub-bin time axis.

    Input `x` has shape `[num_bins, num_sub_bins, num_neurons]`.
    Output has shape `[num_bins, (k_history + 1) * num_sub_bins, num_neurons]`,
    where row t is the temporal concatenation `[x[t-k], ..., x[t-1], x[t]]`.

    Bins that fall within `k_history` of the start of any split (passed in
    `split_starts`) get their cross-boundary sub-bins zeroed so val/test
    rows never carry spikes from the previous split. For bins before the
    first valid history index, missing context is also zeroed.
    """
    if k_history < 0:
        raise ValueError(f"k_history must be >= 0, got {k_history}")
    num_bins, S, N = x.shape
    if k_history == 0:
        return x.copy()
    out = np.zeros((num_bins, (k_history + 1) * S, N), dtype=np.float32)
    for t in range(num_bins):
        # The current bin is the last block.
        out[t, k_history * S : (k_history + 1) * S] = x[t]
        # Fill earlier history slots: index t - k for k = 1..k_history (oldest first).
        for k in range(1, k_history + 1):
            src = t - k
            if src < 0:
                continue  # zero pad at the very start of the recording
            # If src is in a different split than t, zero this block.
            if split_starts is not None:
                # Find the split index of t (the largest start <= t) and of src.
                t_split = max((s for s in split_starts if s <= t), default=0)
                src_split = max((s for s in split_starts if s <= src), default=0)
                if t_split != src_split:
                    continue
            slot = (k_history - k) * S
            out[t, slot : slot + S] = x[src]
    return out


class TrainedLatencySNN:
    """Trainable LIF + linear readout, supervised on 2D velocity.

    Parameters
    ----------
    num_neurons : int
    hidden_dim : int
    tau_ms : float
        Membrane time constant (continuous-time leak).
    threshold : float
        Firing threshold (also subtracted on hard reset).
    bin_size_ms : int
    num_sub_bins : int
        Number of equal-width slots inside each input bin.
    k_history : int
        Number of previous bins to stack as input context. 0 = single bin.
    lr : float
    weight_decay : float
    epochs : int
    patience : int
        Early-stopping patience on val joint R^2.
    batch_size : int
        0 = full batch.
    seed : int
    """

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int = 128,
        tau_ms: float = 10.0,
        threshold: float = 0.30,
        bin_size_ms: int = 50,
        num_sub_bins: int = 10,
        k_history: int = 0,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        patience: int = 15,
        batch_size: int = 0,
        surrogate_slope: float = 25.0,
        seed: int = 0,
    ):
        if num_sub_bins < 1:
            raise ValueError(f"num_sub_bins must be >= 1, got {num_sub_bins}")
        if k_history < 0:
            raise ValueError(f"k_history must be >= 0, got {k_history}")
        self.num_neurons = int(num_neurons)
        self.hidden_dim = int(hidden_dim)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.bin_size_ms = int(bin_size_ms)
        self.num_sub_bins = int(num_sub_bins)
        self.k_history = int(k_history)
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

        x : [B, S, N]   B bins, S total sub-bins (incl. history), N neurons
        W : [hidden, N]
        returns z : [B, hidden] per-bin total hidden spike counts.
        """
        B, S, N = x.shape
        H = W.shape[0]
        decay = float(np.exp(-(self.bin_size_ms / self.num_sub_bins) / self.tau_ms))
        u = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        injections = torch.einsum("bsn,hn->bsh", x, W)
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

        # Pre-bucket events once, then stack history.
        x_all = _sparse_events_to_subbin_counts(
            event_times, event_neurons, self.num_neurons,
            self.bin_size_ms, self.num_sub_bins,
        )
        split_starts: tuple[int, ...] = (
            int(train_idx.min()), int(val_idx.min()),
        )
        x_all = _stack_history(x_all, self.k_history, split_starts)
        logger.info(
            "trained_snn: subbin counts shape=%s k_history=%d -> seq_len=%d",
            x_all.shape, self.k_history, x_all.shape[1],
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

    @property
    def W(self) -> np.ndarray | None:
        return self._W.detach().numpy() if self._W is not None else None

    @property
    def W_out(self) -> np.ndarray | None:
        return self._W_out.detach().numpy() if self._W_out is not None else None

    @property
    def b_out(self) -> np.ndarray | None:
        return self._b_out.detach().numpy() if self._b_out is not None else None

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
        split_starts: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        if self._W is None or self._W_out is None or self._b_out is None:
            raise RuntimeError("TrainedLatencySNN.predict() called before fit()")
        x_all = _sparse_events_to_subbin_counts(
            event_times, event_neurons, self.num_neurons,
            self.bin_size_ms, self.num_sub_bins,
        )
        # If caller didn't pass split_starts, treat the requested idx as a
        # single contiguous slice (no internal split boundary).
        if split_starts is None:
            split_starts = (int(np.min(idx)),)
        x_all = _stack_history(x_all, self.k_history, split_starts)
        x = torch.from_numpy(x_all[idx])
        with torch.no_grad():
            z = self._encode(x, self._W)
            y_pred = z @ self._W_out.T + self._b_out
        return y_pred.numpy()
