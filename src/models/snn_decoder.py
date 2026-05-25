"""Sparse spike-latency LIF SNN decoder. Owned by snn-latency-decoder.

Input: per-bin sparse spike events (event_times, event_neurons) — same schema
as docs/data_interface.md. Output: predicted 2D cursor velocity per bin.
"""
from __future__ import annotations

import numpy as np


class SparseLatencySNN:
    """LIF hidden layer driven by sparse within-bin spike events.

    The within-bin temporal order of events drives the membrane state, so
    permuting that order (the order-shuffle control) should hurt R² if
    spike order carries information.
    """

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int = 128,
        beta: float = 0.9,
        threshold: float = 1.0,
        seed: int = 0,
    ):
        self.num_neurons = num_neurons
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.threshold = threshold
        self.seed = seed

    def fit(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        velocity: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> "SparseLatencySNN":

        num_bins = len(event_times)
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(loc=0, scale=1/np.sqrt(self.num_neurons), size=(self.hidden_dim, self.num_neurons))
        Z = np.zeros((num_bins, self.hidden_dim), np.float32)
        
        for b in range(num_bins):
            u = np.zeros(self.hidden_dim, np.float32)
            spikes = event_neurons[b]
            for neuron in spikes:
                u =  self.beta * u + self.W[:, neuron]
                u[u >= self.threshold] -= self.threshold
            Z[b] = u

        self.readout = LinearReadout()
        self.readout.fit(Z[train_idx], velocity[train_idx])
        
        return self

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
    ) -> np.ndarray:

        
        
        raise NotImplementedError
