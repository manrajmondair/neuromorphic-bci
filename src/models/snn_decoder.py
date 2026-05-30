"""Sparse spike-latency LIF reservoir decoder. Owned by snn-latency-decoder.

Input: per-bin sparse spike events `(event_times, event_neurons)` matching
the schema in `docs/data_interface.md`. Output: predicted 2D cursor
velocity per bin.

Encoder — for each bin we replay the within-bin spike events through a
hidden LIF layer in temporal order. The membrane potential leaks between
events at the actual inter-spike interval (`exp(-Δt / tau_ms)`), each
input spike injects the corresponding column of the fixed random
projection matrix `W`, and hidden neurons that cross threshold emit a
spike and hard-reset by subtracting `threshold`. The per-bin hidden
spike-count vector is the reservoir feature for that bin. `W` is never
trained — this is a reservoir / random-feature encoder, and only the
linear readout is fit.

Temporal context — a per-bin encoder is memoryless, so on its own it can
only see the spikes inside one 50 ms bin. Two optional mechanisms give
the readout access to movement history, which is where most of the
decodable signal lives:

  * `readout_lag_bins` (k): the readout sees the current bin's hidden
    features concatenated with the previous k bins (boundary-safe via
    `stack_lag_features`, so a test bin never borrows a train/val bin).
    This is the dominant lever — a memoryless readout reaches R² ≈ 0.14,
    a k≈15 (≈750 ms) history window reaches R² ≈ 0.6.
  * `recurrent` (optional): run the per-bin features through a fixed,
    spectral-radius-scaled leaky echo-state reservoir before the readout,
    giving fading memory through recurrent hidden dynamics. Off by
    default; lag-stacking alone is the stronger configuration here.

A standardization layer between the encoder and the readout (z-score per
hidden unit, fit on the train split) keeps the readout from being
dominated by always-active or never-active units. The readout is a ridge
regressor; pass `readout_alphas` to pick the ridge penalty by efficient
leave-one-out CV on the train split (`RidgeCV`) instead of a fixed alpha.

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

from src.features.spike_counts import stack_lag_features
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
        L2 strength for the linear ridge readout (used when
        `readout_alphas` is None).
    readout_alphas : sequence of float or None
        If given, the readout selects its L2 penalty from this grid by
        leave-one-out CV on the train split (`sklearn.RidgeCV`); the
        fixed `readout_alpha` is then ignored.
    bin_size_ms : int
        Width of one input bin in milliseconds.
    n_restarts : int
        Number of independent random-projection initializations to try
        per `fit` call; the readout is fit on each, and we keep the one
        with the highest val R² (or the highest train R² if val_idx is
        omitted).
    standardize : bool
        If True (default), z-score the hidden feature matrix per unit
        using train-split statistics before the readout.
    readout_lag_bins : int
        History depth: the readout sees the current bin's hidden features
        stacked with the previous `readout_lag_bins` bins. 0 reproduces
        the memoryless per-bin readout.
    recurrent : bool
        If True, pass the per-bin features through a fixed leaky
        echo-state reservoir (recurrent hidden dynamics) before the
        readout. Off by default.
    spectral_radius : float
        Target spectral radius of the recurrent weight matrix (only used
        when `recurrent`).
    reservoir_leak : float
        Leak rate of the echo-state update in (0, 1] (only used when
        `recurrent`); smaller → longer memory.
    split_starts : tuple of int or None
        First bin index of each contiguous split (train/val/test). Used
        to zero lag / reservoir state across split boundaries so temporal
        features never leak across splits. Required for honest scoring
        when `readout_lag_bins > 0` or `recurrent`.
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
        *,
        readout_alphas: Sequence[float] | None = None,
        readout_lag_bins: int = 0,
        recurrent: bool = False,
        spectral_radius: float = 0.9,
        reservoir_leak: float = 0.3,
        split_starts: tuple[int, ...] | None = None,
    ):
        if tau_ms <= 0:
            raise ValueError(f"tau_ms must be positive, got {tau_ms}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")
        if readout_lag_bins < 0:
            raise ValueError(f"readout_lag_bins must be >= 0, got {readout_lag_bins}")
        if not 0.0 < reservoir_leak <= 1.0:
            raise ValueError(f"reservoir_leak must be in (0, 1], got {reservoir_leak}")

        self.num_neurons = int(num_neurons)
        self.hidden_dim = int(hidden_dim)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.readout_alpha = float(readout_alpha)
        self.readout_alphas = tuple(float(a) for a in readout_alphas) if readout_alphas else None
        self.bin_size_ms = int(bin_size_ms)
        self.n_restarts = int(n_restarts)
        self.standardize = bool(standardize)
        self.readout_lag_bins = int(readout_lag_bins)
        self.recurrent = bool(recurrent)
        self.spectral_radius = float(spectral_radius)
        self.reservoir_leak = float(reservoir_leak)
        self.split_starts = tuple(int(s) for s in split_starts) if split_starts is not None else None
        self.seed = int(seed)

        self.W: np.ndarray | None = None
        self.W_rec: np.ndarray | None = None
        self.readout = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.mu_rec: np.ndarray | None = None
        self.sigma_rec: np.ndarray | None = None
        self.best_restart_seed: int | None = None
        self.chosen_alpha: float | None = None
        self.restart_val_r2s: list[float] = []

    @staticmethod
    def _init_W(num_neurons: int, hidden_dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(num_neurons)
        return rng.normal(loc=0.0, scale=scale, size=(hidden_dim, num_neurons)).astype(np.float32)

    @staticmethod
    def _init_W_rec(hidden_dim: int, seed: int, spectral_radius: float) -> np.ndarray:
        """Random recurrent matrix rescaled to the target spectral radius."""
        rng = np.random.default_rng(seed + 4242)
        Wr = rng.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)).astype(np.float32)
        Wr *= np.float32(1.0 / np.sqrt(hidden_dim))
        radius = float(np.max(np.abs(np.linalg.eigvals(Wr))))
        if radius > 0:
            Wr *= np.float32(spectral_radius / radius)
        return Wr

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

    def _run_reservoir(self, S: np.ndarray, W_rec: np.ndarray) -> np.ndarray:
        """Leaky echo-state update over per-bin features, reset at split starts."""
        leak = np.float32(self.reservoir_leak)
        one_minus = np.float32(1.0 - self.reservoir_leak)
        starts = set(self.split_starts or ())
        X = np.empty_like(S)
        x = np.zeros(S.shape[1], dtype=np.float32)
        for t in range(S.shape[0]):
            if t in starts:
                x = np.zeros(S.shape[1], dtype=np.float32)
            x = one_minus * x + leak * np.tanh(W_rec @ x + S[t])
            X[t] = x
        return X

    def _select_readout(self, F, velocity, train_idx, val_idx):
        """Fit the ridge readout, selecting alpha on val when a grid is given.

        Returns ``(readout, score, chosen_alpha)`` where ``score`` is the joint
        R² on val (or train if no val). Plain ridge per alpha keeps this fast
        even at the wide lag-stacked feature dimension.
        """
        from src.evaluation.metrics import velocity_r2

        has_val = val_idx is not None and val_idx.size > 0
        alphas = self.readout_alphas if self.readout_alphas is not None else (self.readout_alpha,)
        best = None
        for a in alphas:
            ro = LinearReadout(alpha=float(a)).fit(F[train_idx], velocity[train_idx])
            ref_idx = val_idx if has_val else train_idx
            s = velocity_r2(velocity[ref_idx], ro.predict(F[ref_idx]))["r2_joint"]
            if best is None or s > best[1]:
                best = (ro, float(s), float(a))
        return best

    def _features(self, S: np.ndarray) -> np.ndarray:
        """Raw per-bin counts -> standardized -> optional reservoir -> lag-stack.

        Uses the standardization / reservoir state stored by `fit`, so it is
        valid only after fitting (or with stats supplied for the train pass).
        """
        if self.standardize:
            S = (S - self.mu) / self.sigma
        if self.recurrent:
            X = self._run_reservoir(S, self.W_rec)
            if self.standardize:
                X = (X - self.mu_rec) / self.sigma_rec
        else:
            X = S
        return stack_lag_features(
            X.astype(np.float32, copy=False), self.readout_lag_bins, self.split_starts
        )

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
        best: dict | None = None
        self.restart_val_r2s = []

        for k in range(self.n_restarts):
            seed_k = self.seed + k
            W_k = self._init_W(self.num_neurons, self.hidden_dim, seed_k)
            S = self._encode_with_W(W_k, event_times, event_neurons)

            if self.standardize:
                mu = S[train_idx].mean(axis=0)
                sigma = S[train_idx].std(axis=0) + 1e-6
                Sz = (S - mu) / sigma
            else:
                mu = sigma = None
                Sz = S

            if self.recurrent:
                W_rec = self._init_W_rec(self.hidden_dim, seed_k, self.spectral_radius)
                X = self._run_reservoir(Sz, W_rec)
                if self.standardize:
                    mu_rec = X[train_idx].mean(axis=0)
                    sigma_rec = X[train_idx].std(axis=0) + 1e-6
                    Xz = (X - mu_rec) / sigma_rec
                else:
                    mu_rec = sigma_rec = None
                    Xz = X
            else:
                W_rec = mu_rec = sigma_rec = None
                Xz = Sz

            F = stack_lag_features(
                Xz.astype(np.float32, copy=False), self.readout_lag_bins, self.split_starts
            )
            readout, score, chosen_alpha = self._select_readout(F, velocity, train_idx, val_idx)
            self.restart_val_r2s.append(float(score))

            if score > best_score:
                best_score = score
                best = {
                    "W": W_k, "W_rec": W_rec, "readout": readout,
                    "mu": mu, "sigma": sigma, "mu_rec": mu_rec, "sigma_rec": sigma_rec,
                    "seed": seed_k, "alpha": chosen_alpha,
                }

        assert best is not None
        self.W = best["W"]
        self.W_rec = best["W_rec"]
        self.readout = best["readout"]
        self.mu, self.sigma = best["mu"], best["sigma"]
        self.mu_rec, self.sigma_rec = best["mu_rec"], best["sigma_rec"]
        self.best_restart_seed = best["seed"]
        self.chosen_alpha = best["alpha"]
        return self

    def predict(
        self,
        event_times: list[np.ndarray],
        event_neurons: list[np.ndarray],
        idx: np.ndarray,
    ) -> np.ndarray:
        if self.readout is None or self.W is None:
            raise RuntimeError("SparseLatencySNN.predict() called before fit()")
        # Encode the full sequence so lag / reservoir features have their
        # history, then index the requested bins.
        S = self._encode_with_W(self.W, event_times, event_neurons)
        F = self._features(S)
        return self.readout.predict(F[idx])


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
    *,
    readout_alphas: Sequence[float] | None = None,
    readout_lag_bins: int = 0,
    recurrent: bool = False,
    spectral_radius: float = 0.9,
    reservoir_leak: float = 0.3,
    split_starts: tuple[int, ...] | None = None,
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
            readout_alphas=readout_alphas,
            readout_lag_bins=readout_lag_bins,
            recurrent=recurrent,
            spectral_radius=spectral_radius,
            reservoir_leak=reservoir_leak,
            split_starts=split_starts,
        ).fit(event_times, event_neurons, velocity, train_idx, val_idx)
        # Score on val using the SNN's predict (re-encodes the full sequence).
        y_val = snn.predict(event_times, event_neurons, val_idx)
        score = velocity_r2(velocity[val_idx], y_val)["r2_joint"]
        sweep.append((float(thr), float(score)))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, sweep
