"""Apply controlled gaussian jitter to within-bin spike times.

The schema requires each bin's `event_times` to be sorted ascending and
strictly within `[0, bin_size_ms)`. After jittering, this routine
re-sorts and clips so the result still satisfies the invariant.

Owned by snn-latency-decoder.
"""
from __future__ import annotations

import numpy as np


def jitter_event_times(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    sigma_ms: float,
    bin_size_ms: int,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Add gaussian noise to spike times, re-sort, clip into the bin.

    Parameters
    ----------
    event_times, event_neurons :
        Per-bin sparse-event lists from the canonical data interface.
    sigma_ms : float
        Standard deviation of the per-spike gaussian jitter in ms.
        sigma_ms = 0 returns the inputs unchanged.
    bin_size_ms : int
        Width of one bin in ms; jittered times are clipped to
        `[0, bin_size_ms - 1e-3]`.
    seed : int
        RNG seed.

    Returns
    -------
    (jittered_event_times, reordered_event_neurons) with the same shapes
    as the inputs but spike timing perturbed and the per-bin order
    re-sorted by the new times.
    """
    if sigma_ms < 0:
        raise ValueError(f"sigma_ms must be non-negative, got {sigma_ms}")
    if sigma_ms == 0:
        return event_times, event_neurons
    rng = np.random.default_rng(seed)
    out_t: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    eps = 1e-4
    upper = float(bin_size_ms) - eps
    for times, neurons in zip(event_times, event_neurons, strict=True):
        n = times.size
        if n == 0:
            out_t.append(times)
            out_n.append(neurons)
            continue
        jitter = rng.normal(scale=sigma_ms, size=n).astype(times.dtype)
        new_t = np.clip(times + jitter, 0.0, upper)
        order = np.argsort(new_t, kind="stable")
        # Re-strict-monotonic: bump duplicates by `eps * rank` so the
        # data-interface invariant still holds.
        sorted_t = new_t[order].astype(np.float64)
        if sorted_t.size > 1:
            sorted_t += eps * np.arange(sorted_t.size, dtype=np.float64)
        out_t.append(sorted_t.astype(times.dtype))
        out_n.append(neurons[order])
    return out_t, out_n
