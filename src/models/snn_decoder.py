"""Sparse spike-latency LIF SNN decoder. Owned by snn-latency-decoder.

Input: per-bin sparse spike events `(event_times, event_neurons)` matching
the schema in `docs/data_interface.md`. Output: predicted 2D cursor
velocity per bin.

Architecture — for each bin we replay the within-bin spike events through
a hidden LIF layer in temporal order. The membrane potential leaks
between events at the actual inter-spike interval (`exp(-Δt / tau_ms)`),
each input spike injects the corresponding column of the random
projection matrix `W`, and hidden neurons that cross threshold emit a
spike and hard-reset by subtracting `threshold`. The per-bin hidden
spike-count vector is the feature passed to a ridge readout that maps
to 2D velocity.

A standardization layer between the encoder and the readout (z-score per
hidden unit, fit on the train split) keeps the readout from being
dominated by always-active or never-active units.

`tune_threshold_on_val` selects the threshold that maximises the joint
R² on the val split for a given event budget — sparser budgets benefit
from lower thresholds because there are fewer input events to push the
membrane.

The encoder is vectorized over hidden_dim and across the events within a
bin via a numpy-friendly per-bin loop with one matrix-vector update per
event. Bin loop is Python (events are jagged across bins), but each
event step is O(hidden_dim) numpy ops with no python-level branching.

Within-bin order changes which neurons accumulate input together; spike
latency changes how much the membrane has decayed between events. The
order-shuffle control in `src.controls.order_shuffle` keeps the same
events but permutes their `(time, neuron)` pairs, ablating order while
leaving the same overall mean leak.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.models.readout import LinearReadout


class SparseLatencySNN:
    """LIF hidden layer driven by sparse within-bin spike events.

    Parameters
    ----------
    num_neurons : int
        Number of input neurons.
    hidden_dim : int
        Width of the hidden LIF layer.
    tau_ms : float
        Membrane time constant in milliseconds.
    threshold : float
        Firing threshold. Lower threshold → denser hidden firing.
    readout_alpha : float
        L2 strength for the linear ridge readout.
    bin_size_ms : int
        Width of one input bin in milliseconds.
    n_restarts : int
        Number of independent random-projection initializations to try
        per `fit` call; the readout is fit on each, and we keep the one
        with the highest val R² (or the highest train R² if val_idx is
        omitted).
    standardize : bool
        If True (default), z-score the hidden spike-count matrix per
        unit using train-split statistics before fitting the readout.
    seed : int
        Random seed; restarts use `seed + k` for k=0..n_restarts-1.
    """

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int = 256,
        tau_ms: float = 10.0,
        threshold: float = 0.5,
        readout_alpha: float = 1.0,
        bin_size_ms: int = 50,
        n_restarts: int = 3,
        standardize: bool = True,
        seed: int = 0,
    ):
        if tau_ms <= 0:
            raise ValueError(f"tau_ms must be positive, got {tau_ms}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

        self.num_neurons = int(num_neurons)
        self.hidden_dim = int(hidden_dim)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.readout_alpha = float(readout_alpha)
        self.bin_size_ms = int(bin_size_ms)
        self.n_restarts = int(n_restarts)
        self.standardize = bool(standardize)
        self.seed = int(seed)

        self.W: np.ndarray | None = None
        self.readout: LinearReadout | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.best_restart_seed: int | None = None
        self.restart_val_r2s: list[float] = []

    @staticmethod
    def _init_W(num_neurons: int, hidden_dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(num_neurons)
        return rng.normal(loc=0.0, scale=scale, size=(hidden_dim, num_neurons)).astype(np.float32)

    def _encode_with_W(
        self,
        W: np.ndarray,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
    ) -> np.ndarray:
        num_bins = len(event_neurons)
        Z = np.zeros((num_bins, self.hidden_dim), dtype=np.float32)
        tau = self.tau_ms
        thr = self.threshold
        for t in range(num_bins):
            times = event_times[t]
            neurons = event_neurons[t]
            n_ev = neurons.size
            if n_ev == 0:
                continue
            u = np.zeros(self.hidden_dim, dtype=np.float32)
            spike_count = np.zeros(self.hidden_dim, dtype=np.float32)
            last_t = 0.0
            decays = np.exp(-(np.diff(times, prepend=np.float32(0.0))) / tau).astype(np.float32)
            for k in range(n_ev):
                if k > 0 or times[0] > 0:
                    u *= decays[k]
                u += W[:, int(neurons[k])]
                fired = u >= thr
                if fired.any():
                    spike_count[fired] += 1.0
                    u[fired] -= thr
                last_t = float(times[k])  # noqa: F841 — kept for clarity / future use
            Z[t] = spike_count
        return Z

    def _encode(self, event_times, event_neurons):
        if self.W is None:
            raise RuntimeError("SparseLatencySNN._encode() called before fit()")
        return self._encode_with_W(self.W, event_times, event_neurons)

    def fit(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        velocity: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> "SparseLatencySNN":
        from src.evaluation.metrics import velocity_r2

        best_score = -np.inf
        best_W: np.ndarray | None = None
        best_readout: LinearReadout | None = None
        best_mu = None
        best_sigma = None
        best_seed = None
        self.restart_val_r2s = []

        for k in range(self.n_restarts):
            seed_k = self.seed + k
            W_k = self._init_W(self.num_neurons, self.hidden_dim, seed_k)
            Z = self._encode_with_W(W_k, event_times, event_neurons)
            if self.standardize:
                mu = Z[train_idx].mean(axis=0)
                sigma = Z[train_idx].std(axis=0) + 1e-6
                Z_std = (Z - mu) / sigma
            else:
                mu = sigma = None
                Z_std = Z
            readout = LinearReadout(alpha=self.readout_alpha).fit(
                Z_std[train_idx], velocity[train_idx]
            )
            if val_idx is not None and val_idx.size > 0:
                y_val = readout.predict(Z_std[val_idx])
                score = velocity_r2(velocity[val_idx], y_val)["r2_joint"]
            else:
                y_train = readout.predict(Z_std[train_idx])
                score = velocity_r2(velocity[train_idx], y_train)["r2_joint"]
            self.restart_val_r2s.append(float(score))
            if score > best_score:
                best_score = score
                best_W = W_k
                best_readout = readout
                best_mu, best_sigma = mu, sigma
                best_seed = seed_k

        self.W = best_W
        self.readout = best_readout
        self.mu = best_mu
        self.sigma = best_sigma
        self.best_restart_seed = best_seed
        return self

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
    ) -> np.ndarray:
        if self.readout is None or self.W is None:
            raise RuntimeError("SparseLatencySNN.predict() called before fit()")
        times = [event_times[i] for i in idx]
        neurons = [event_neurons[i] for i in idx]
        Z = self._encode_with_W(self.W, times, neurons)
        if self.standardize and self.mu is not None and self.sigma is not None:
            Z = (Z - self.mu) / self.sigma
        return self.readout.predict(Z)


def tune_threshold_on_val(
    num_neurons: int,
    bin_size_ms: int,
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    velocity: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    thresholds: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00),
    hidden_dim: int = 256,
    tau_ms: float = 10.0,
    readout_alpha: float = 1.0,
    n_restarts: int = 1,
    standardize: bool = True,
    seed: int = 0,
) -> tuple[float, list[tuple[float, float]]]:
    """Sweep thresholds, fit the SNN at each, return (best_threshold, sweep)."""
    from src.evaluation.metrics import velocity_r2

    sweep: list[tuple[float, float]] = []
    best_thr = float(thresholds[0])
    best_score = -np.inf
    for thr in thresholds:
        snn = SparseLatencySNN(
            num_neurons=num_neurons,
            hidden_dim=hidden_dim,
            tau_ms=tau_ms,
            threshold=float(thr),
            readout_alpha=readout_alpha,
            bin_size_ms=bin_size_ms,
            n_restarts=n_restarts,
            standardize=standardize,
            seed=seed,
        ).fit(event_times, event_neurons, velocity, train_idx, val_idx)
        # Score on val using the SNN's predict (re-encodes the val bins for safety).
        y_val = snn.predict(event_times, event_neurons, val_idx)
        score = velocity_r2(velocity[val_idx], y_val)["r2_joint"]
        sweep.append((float(thr), float(score)))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, sweep
