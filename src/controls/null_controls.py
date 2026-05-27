"""Null-control transforms for the multi-shuffle statistical battery.

Each function takes (event_times, event_neurons) and returns a perturbed
version that destroys a specific kind of structure while keeping others.
Used to build null distributions in `scripts/run_null_battery.py`.

Available controls:
  * `shuffle_within_bin_order` (lives in order_shuffle.py): permute the
    (time, neuron) pairs inside each bin. Kills order, keeps rates.
  * `phase_randomize_times` : draw fresh uniform-in-[0, bin_size_ms)
    timestamps for every spike, keep neuron identities and per-bin counts.
    Kills any precise within-bin timing, keeps rates.
  * `shuffle_neuron_identities` : permute the neuron axis so the
    population's spike-time pattern is preserved but each spike is
    attributed to a random neuron. Kills neuron-specific tuning.
  * `circular_shift_bins` : circularly roll the entire dataset's per-bin
    event lists by a random offset. Kills bin-to-velocity alignment,
    keeps every other property exactly.
"""
from __future__ import annotations

import numpy as np


def phase_randomize_times(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    bin_size_ms: int,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Replace within-bin spike times with fresh uniform draws; preserve neuron ids.

    Re-sorts the events by the new times so the data-interface invariant
    holds. Adds a tiny rank-based eps to keep strict monotonicity.
    """
    rng = np.random.default_rng(seed)
    eps = 1e-4
    out_t: list[np.ndarray] = []
    out_n: list[np.ndarray] = []
    for times, neurons in zip(event_times, event_neurons, strict=True):
        n = times.size
        if n == 0:
            out_t.append(times)
            out_n.append(neurons)
            continue
        upper = float(bin_size_ms) - eps * n
        new_t = rng.uniform(0.0, upper, size=n).astype(times.dtype)
        order = np.argsort(new_t, kind="stable")
        sorted_t = new_t[order].astype(np.float64)
        if sorted_t.size > 1:
            sorted_t += eps * np.arange(sorted_t.size, dtype=np.float64)
        out_t.append(sorted_t.astype(times.dtype))
        out_n.append(neurons[order])
    return out_t, out_n


def shuffle_neuron_identities(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    num_neurons: int,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Permute the neuron-id axis dataset-wide.

    The same permutation is applied to every bin, so the population's
    cross-bin spike-time pattern is preserved but each neuron's identity
    (and thus its preferred direction in the trained decoder) is
    scrambled. This isolates neuron-specific tuning as a contributor to
    decoding R².
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_neurons).astype(np.int32)
    out_n = [perm[neurons] if neurons.size else neurons for neurons in event_neurons]
    return list(event_times), out_n


def circular_shift_bins(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    seed: int = 0,
    min_shift_bins: int = 100,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Roll the entire dataset of per-bin event lists by a random offset.

    Keeps every per-bin event distribution intact but decorrelates bins
    from their original velocity targets. Useful when the user wants
    a null where rate, timing, and identity are all preserved.

    `min_shift_bins` prevents the trivial near-identity shift.
    """
    num_bins = len(event_times)
    if num_bins == 0:
        return list(event_times), list(event_neurons)
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(min_shift_bins, num_bins - min_shift_bins))
    out_t = [event_times[(t - shift) % num_bins] for t in range(num_bins)]
    out_n = [event_neurons[(t - shift) % num_bins] for t in range(num_bins)]
    return out_t, out_n
