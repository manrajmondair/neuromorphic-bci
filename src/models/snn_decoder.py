"""Sparse spike-latency LIF SNN decoder. Owned by snn-latency-decoder.

Input: per-bin sparse spike events `(event_times, event_neurons)` matching the
schema in `docs/data_interface.md`. Output: predicted 2D cursor velocity per
bin.

Architecture — for each 50 ms bin we replay the within-bin spike events
through a hidden LIF layer in temporal order. The membrane potential leaks
between events at the actual inter-spike interval (`exp(-dt / tau_ms)`), each
input spike injects the corresponding column of the random projection
matrix `W`, and hidden neurons that cross threshold emit a spike and
hard-reset by subtracting `threshold`. The per-bin hidden spike count
vector is the feature passed to a linear ridge readout that maps to 2D
velocity.

Because the leak is keyed to real spike times, both *order* and *latency*
information affect the output:
  * Within-bin order changes which neurons accumulate input together.
  * Within-bin latency changes how much the membrane has decayed between
    events.

The order-shuffle control in `src.controls.order_shuffle` keeps the same
events but permutes their `(time, neuron)` pairs — so it ablates only
order, leaving the same overall mean leak; comparing its R^2 against the
real-order SNN isolates the contribution of within-bin order.
"""
from __future__ import annotations

import numpy as np

from src.models.readout import LinearReadout


class SparseLatencySNN:
    """LIF hidden layer driven by sparse within-bin spike events.

    Parameters
    ----------
    num_neurons : int
        Number of input neurons (matches `processed["num_neurons"]`).
    hidden_dim : int
        Width of the hidden LIF layer.
    tau_ms : float
        Membrane time constant in milliseconds. Smaller -> more sensitive to
        spike timing.
    threshold : float
        Firing threshold. With weights drawn at scale `1/sqrt(num_neurons)`
        this controls the per-bin spike count of the hidden layer.
    readout_alpha : float
        L2 strength for the linear readout fit by ridge.
    bin_size_ms : int
        Width of one input bin in milliseconds. Used to flush membrane state
        at the end of each bin.
    seed : int
        Random seed for the input projection matrix.
    """

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int = 128,
        tau_ms: float = 10.0,
        threshold: float = 0.5,
        readout_alpha: float = 1.0,
        bin_size_ms: int = 50,
        seed: int = 0,
    ):
        if tau_ms <= 0:
            raise ValueError(f"tau_ms must be positive, got {tau_ms}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.num_neurons = int(num_neurons)
        self.hidden_dim = int(hidden_dim)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.readout_alpha = float(readout_alpha)
        self.bin_size_ms = int(bin_size_ms)
        self.seed = int(seed)

        self.W: np.ndarray | None = None
        self.readout: LinearReadout | None = None

    def _encode(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
    ) -> np.ndarray:
        """Run continuous-time LIF over each bin and return [num_bins, hidden] spike counts."""
        if self.W is None:
            raise RuntimeError("SparseLatencySNN._encode() called before fit()")
        num_bins = len(event_neurons)
        Z = np.zeros((num_bins, self.hidden_dim), dtype=np.float32)
        tau = self.tau_ms
        thr = self.threshold

        for t in range(num_bins):
            times = event_times[t]
            neurons = event_neurons[t]
            u = np.zeros(self.hidden_dim, dtype=np.float32)
            spike_count = np.zeros(self.hidden_dim, dtype=np.float32)
            last_t = 0.0
            for k in range(neurons.size):
                dt = float(times[k]) - last_t
                # Leaky decay between events at the actual inter-spike interval.
                if dt > 0:
                    u *= np.exp(-dt / tau)
                u += self.W[:, int(neurons[k])]
                # Emit spike + hard reset (subtract threshold).
                fired = u >= thr
                if fired.any():
                    spike_count[fired] += 1.0
                    u[fired] -= thr
                last_t = float(times[k])
            Z[t] = spike_count
        return Z

    def fit(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        velocity: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> SparseLatencySNN:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.num_neurons)
        self.W = rng.normal(
            loc=0.0, scale=scale, size=(self.hidden_dim, self.num_neurons)
        ).astype(np.float32)

        Z = self._encode(event_times, event_neurons)
        self.readout = LinearReadout(alpha=self.readout_alpha).fit(
            Z[train_idx], velocity[train_idx]
        )
        # val_idx is kept in the signature for parity with sklearn-style
        # interfaces; no validation-driven hyperparameter selection happens
        # here because the SNN configuration is fixed at construction time.
        return self

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
    ) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("SparseLatencySNN.predict() called before fit()")
        times = [event_times[i] for i in idx]
        neurons = [event_neurons[i] for i in idx]
        Z = self._encode(times, neurons)
        return self.readout.predict(Z)
