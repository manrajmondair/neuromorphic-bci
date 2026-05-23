"""Build the processed-data object defined in docs/data_interface.md.

Pipeline (in order, with shape logging at each step):

    1. Open NLB MC_RTT NWB file at native (1 ms) resolution.
    2. Extract a [T_native, num_neurons] spike-count matrix and a
       [T_native, 2] cursor trace.
    3. Bin spikes into `bin_size_ms` windows, producing both a dense
       `spike_counts` matrix and per-bin sparse `(event_times, event_neurons)`
       lists with within-bin times in ms.
    4. Subsample cursor to bin starts → `cursor_pos[num_bins, 2]`.
    5. Compute 2D `velocity` via central (default) or forward finite
       differences.
    6. Build a deterministic time-contiguous train/val/test split with a
       `boundary_gap` of unused bins on each side of every split boundary
       so finite-difference velocities cannot leak labels across splits.
    7. Validate every invariant from `docs/data_interface.md` before
       returning.

Everything is logged so the shape, dtype, and count of every intermediate
array is visible in the script's stdout.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .load_nlb import find_nwb_file, load_mc_rtt_dataset
from .split import time_contiguous_split

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """All knobs for the preprocessing pipeline in one place."""

    raw_dir: Path = Path("data/raw")
    bin_size_ms: int = 50
    split_fracs: tuple[float, float, float] = (0.70, 0.15, 0.15)
    boundary_gap: int = 1
    velocity_mode: str = "central"
    behavior_signal: str = "cursor_pos"

    def __post_init__(self) -> None:
        if self.bin_size_ms <= 0:
            raise ValueError(f"bin_size_ms must be positive, got {self.bin_size_ms}")
        if abs(sum(self.split_fracs) - 1.0) > 1e-6:
            raise ValueError(f"split_fracs must sum to 1.0, got {self.split_fracs}")
        if self.velocity_mode not in {"central", "forward"}:
            raise ValueError(f"velocity_mode must be 'central' or 'forward', got {self.velocity_mode}")


# -------------------------------------------------------------------------
# Building blocks
# -------------------------------------------------------------------------


def _pull_signal(dataset: Any, name: str) -> np.ndarray | None:
    """Return a [T_native, num_channels] np.ndarray for `name`, or None if absent."""
    try:
        top_level = dataset.data.columns.get_level_values(0)
    except Exception:  # noqa: BLE001
        return None
    if name not in top_level:
        return None
    arr = dataset.data[name].to_numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _extract_spike_events(
    spikes_native: np.ndarray,
    native_bin_ms: float,
    target_bin_ms: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Bin a native-resolution spike-count matrix into target_bin_ms windows.

    Produces a dense [num_bins, num_neurons] count matrix AND per-bin sparse
    event lists with within-bin times in ms (sorted ascending, strictly
    monotonic — collisions at the same native ms are broken with sub-ms
    jitter so downstream consumers can rely on `np.all(np.diff(times) > 0)`).
    """
    if spikes_native.ndim != 2:
        raise ValueError(f"spikes_native must be 2-D, got shape {spikes_native.shape}")
    T_native, N = spikes_native.shape
    if T_native == 0 or N == 0:
        raise ValueError(f"empty spike matrix: shape {spikes_native.shape}")

    bins_per_target = int(round(target_bin_ms / native_bin_ms))
    if bins_per_target < 1 or abs(bins_per_target * native_bin_ms - target_bin_ms) > 1e-6:
        raise ValueError(
            f"target_bin_ms={target_bin_ms} is not a positive multiple of "
            f"native_bin_ms={native_bin_ms}"
        )
    num_bins = T_native // bins_per_target
    dropped = T_native - num_bins * bins_per_target
    logger.info(
        "extract_spike_events: native_bin=%g ms, target_bin=%d ms, bins_per_target=%d, num_bins=%d (dropped %d trailing native samples)",
        native_bin_ms,
        target_bin_ms,
        bins_per_target,
        num_bins,
        dropped,
    )

    spikes_trim = spikes_native[: num_bins * bins_per_target].astype(np.int32, copy=False)
    # Reshape into [num_bins, bins_per_target, num_neurons] for vectorized counts
    reshaped = spikes_trim.reshape(num_bins, bins_per_target, N)
    spike_counts = reshaped.sum(axis=1).astype(np.int32)
    logger.info(
        "spike_counts: shape=%s dtype=%s total=%d mean/bin=%.2f",
        spike_counts.shape,
        spike_counts.dtype,
        int(spike_counts.sum()),
        float(spike_counts.sum() / max(num_bins, 1)),
    )

    event_times: list[np.ndarray] = []
    event_neurons: list[np.ndarray] = []
    eps = 1e-6  # microsecond-scale jitter for strict monotonic ordering

    for t in range(num_bins):
        bin_block = reshaped[t]  # [bins_per_target, N]
        offsets, neurons = np.nonzero(bin_block)
        if offsets.size == 0:
            event_times.append(np.zeros(0, dtype=np.float32))
            event_neurons.append(np.zeros(0, dtype=np.int32))
            continue
        counts_at = bin_block[offsets, neurons].astype(np.int32)
        # Mid-of-native-bin time in ms, within the target bin
        base = (offsets.astype(np.float64) + 0.5) * native_bin_ms
        if (counts_at > 1).any():
            # Distribute c spikes uniformly inside the native bin to keep ordering deterministic
            expanded_times = []
            expanded_neurons = []
            for o, n, c in zip(offsets, neurons, counts_at):
                if c == 1:
                    expanded_times.append(np.array([(o + 0.5) * native_bin_ms]))
                    expanded_neurons.append(np.array([n], dtype=np.int32))
                else:
                    inner = (np.arange(c) + 0.5) / c
                    expanded_times.append((o + inner) * native_bin_ms)
                    expanded_neurons.append(np.full(c, n, dtype=np.int32))
            times = np.concatenate(expanded_times)
            neurs = np.concatenate(expanded_neurons)
        else:
            times = base
            neurs = neurons.astype(np.int32)

        order = np.argsort(times, kind="stable")
        times_sorted = times[order]
        neurs_sorted = neurs[order]
        # Strict-monotonic tiebreak: nanosecond-scale jitter by within-bin index
        if times_sorted.size > 1:
            times_sorted = times_sorted + eps * np.arange(times_sorted.size, dtype=np.float64)
        event_times.append(times_sorted.astype(np.float32))
        event_neurons.append(neurs_sorted)

    # Spot-check spike_counts vs event_neurons on a sample
    sample = np.linspace(0, num_bins - 1, num=min(num_bins, 50)).astype(int)
    for t in sample:
        from_events = np.bincount(event_neurons[t], minlength=N)
        if not np.array_equal(spike_counts[t], from_events):
            raise ValueError(f"internal: spike_counts and event_neurons disagree at bin {t}")
    logger.info(
        "event lists: %d bins, mean=%.2f events/bin, max=%d, strict-monotonic by construction",
        num_bins,
        float(spike_counts.sum() / max(num_bins, 1)),
        int(spike_counts.sum(axis=1).max()),
    )
    return spike_counts, event_times, event_neurons


def _extract_cursor_position(
    cursor_native: np.ndarray,
    native_bin_ms: float,
    target_bin_ms: int,
    num_bins: int,
) -> np.ndarray:
    """Subsample a [T_native, >=2] cursor trace down to bin-start positions."""
    if cursor_native.ndim != 2 or cursor_native.shape[1] < 2:
        raise ValueError(f"cursor_native must be [T, >=2], got {cursor_native.shape}")
    bins_per_target = int(round(target_bin_ms / native_bin_ms))
    needed = (num_bins - 1) * bins_per_target + 1
    if cursor_native.shape[0] < needed:
        raise ValueError(
            f"cursor too short: need {needed} native samples, got {cursor_native.shape[0]}"
        )
    indices = np.arange(num_bins, dtype=np.int64) * bins_per_target
    cursor_binned = cursor_native[indices, :2].astype(np.float32)
    if not np.isfinite(cursor_binned).all():
        n_bad = int(np.sum(~np.isfinite(cursor_binned)))
        logger.warning(
            "cursor_binned has %d non-finite values; will surface in NaN/Inf checks downstream",
            n_bad,
        )
    logger.info(
        "cursor: native shape=%s -> binned shape=%s, finite=%s",
        cursor_native.shape,
        cursor_binned.shape,
        bool(np.isfinite(cursor_binned).all()),
    )
    return cursor_binned


def _compute_velocity(
    cursor_pos: np.ndarray,
    bin_size_ms: int,
    mode: str = "central",
) -> np.ndarray:
    """Finite-difference velocity at each bin in units / second.

    `central`: v[t] = (pos[t+1] - pos[t-1]) / (2 * dt) for interior bins;
               one-sided forward/backward at the very first/last bin.
    `forward`: v[t] = (pos[t+1] - pos[t]) / dt; last bin repeats v[-2].
    """
    if cursor_pos.shape[1] != 2:
        raise ValueError(f"cursor_pos must have 2 cols, got shape {cursor_pos.shape}")
    if mode not in {"central", "forward"}:
        raise ValueError(f"unknown velocity mode: {mode}")
    if cursor_pos.shape[0] < 2:
        raise ValueError(f"need at least 2 bins to compute velocity, got {cursor_pos.shape[0]}")

    dt_s = bin_size_ms / 1000.0
    v = np.zeros_like(cursor_pos, dtype=np.float32)
    if mode == "central":
        v[1:-1] = (cursor_pos[2:] - cursor_pos[:-2]) / (2.0 * dt_s)
        v[0] = (cursor_pos[1] - cursor_pos[0]) / dt_s
        v[-1] = (cursor_pos[-1] - cursor_pos[-2]) / dt_s
    else:
        v[:-1] = (cursor_pos[1:] - cursor_pos[:-1]) / dt_s
        v[-1] = v[-2]

    speed = np.linalg.norm(v, axis=1)
    logger.info(
        "velocity (%s diff): shape=%s dtype=%s mean|v|=%.4f max|v|=%.4f any_nan=%s",
        mode,
        v.shape,
        v.dtype,
        float(speed.mean()),
        float(speed.max()),
        bool(np.isnan(v).any()),
    )
    return v


def _validate_processed(processed: dict[str, Any]) -> None:
    """Assert every invariant from docs/data_interface.md."""
    spike_counts = processed["spike_counts"]
    event_times = processed["event_times"]
    event_neurons = processed["event_neurons"]
    velocity = processed["velocity"]
    train_idx = processed["train_idx"]
    val_idx = processed["val_idx"]
    test_idx = processed["test_idx"]

    num_bins, num_neurons = spike_counts.shape
    if len(event_times) != num_bins:
        raise AssertionError(f"event_times length {len(event_times)} != num_bins {num_bins}")
    if len(event_neurons) != num_bins:
        raise AssertionError(f"event_neurons length {len(event_neurons)} != num_bins {num_bins}")
    if velocity.shape != (num_bins, 2):
        raise AssertionError(f"velocity shape {velocity.shape} != ({num_bins}, 2)")
    if np.isnan(velocity).any():
        raise AssertionError("velocity contains NaN")

    sample = np.linspace(0, num_bins - 1, num=min(num_bins, 100)).astype(int)
    for t in sample:
        times = event_times[t]
        neurons = event_neurons[t]
        if times.shape != neurons.shape:
            raise AssertionError(f"bin {t}: times {times.shape} vs neurons {neurons.shape}")
        if times.size > 1 and not np.all(np.diff(times) > 0):
            raise AssertionError(f"bin {t}: event_times not strictly monotonic")
        from_events = np.bincount(neurons, minlength=num_neurons)
        if not np.array_equal(spike_counts[t], from_events):
            raise AssertionError(f"bin {t}: spike_counts inconsistent with event_neurons")

    union = np.concatenate([train_idx, val_idx, test_idx])
    if union.size == 0:
        raise AssertionError("all splits empty")
    if len(set(union.tolist())) != union.size:
        raise AssertionError("split indices overlap")
    if union.min() < 0 or union.max() >= num_bins:
        raise AssertionError("split indices out of range")

    logger.info("invariants from docs/data_interface.md: PASS")


# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------


def preprocess_mc_rtt(
    raw_dir: Path | str = Path("data/raw"),
    bin_size_ms: int = 50,
    split_fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
    boundary_gap: int = 1,
    velocity_mode: str = "central",
    behavior_signal: str = "cursor_pos",
    seed: int = 0,  # noqa: ARG001 — kept for API stability; split is deterministic
) -> dict[str, Any]:
    """End-to-end preprocessing — see module docstring for the full pipeline."""
    cfg = PreprocessConfig(
        raw_dir=Path(raw_dir),
        bin_size_ms=bin_size_ms,
        split_fracs=split_fracs,
        boundary_gap=boundary_gap,
        velocity_mode=velocity_mode,
        behavior_signal=behavior_signal,
    )

    logger.info("=" * 72)
    logger.info("preprocess_mc_rtt config: %s", cfg)
    logger.info("=" * 72)

    # 1. Open NWB
    nwb_path = find_nwb_file(cfg.raw_dir)
    dataset = load_mc_rtt_dataset(nwb_path)
    native_bin_ms = float(getattr(dataset, "bin_width", 1.0))
    if native_bin_ms <= 0:
        raise ValueError(f"non-positive native bin_width: {native_bin_ms}")

    # 2. Pull raw signals
    spikes_native = _pull_signal(dataset, "spikes")
    if spikes_native is None:
        raise KeyError("'spikes' signal not found in NWB dataset")
    spikes_native = spikes_native.astype(np.int32, copy=False)
    logger.info(
        "spikes_native: shape=%s dtype=%s total=%d",
        spikes_native.shape,
        spikes_native.dtype,
        int(spikes_native.sum()),
    )

    cursor_native = _pull_signal(dataset, cfg.behavior_signal)
    if cursor_native is None:
        finger = _pull_signal(dataset, "finger_pos")
        if finger is None:
            raise KeyError(
                f"neither '{cfg.behavior_signal}' nor 'finger_pos' present in dataset"
            )
        logger.warning(
            "behavior signal %r not found; falling back to finger_pos[:, :2]",
            cfg.behavior_signal,
        )
        cursor_native = finger[:, :2]
    cursor_native = cursor_native.astype(np.float32, copy=False)
    logger.info("cursor_native: shape=%s dtype=%s", cursor_native.shape, cursor_native.dtype)

    # Crop to common length if signals disagree
    if cursor_native.shape[0] != spikes_native.shape[0]:
        T_common = min(cursor_native.shape[0], spikes_native.shape[0])
        logger.warning(
            "spikes T (%d) != cursor T (%d); cropping both to %d",
            spikes_native.shape[0],
            cursor_native.shape[0],
            T_common,
        )
        spikes_native = spikes_native[:T_common]
        cursor_native = cursor_native[:T_common]

    # 3. Bin spikes
    spike_counts, event_times, event_neurons = _extract_spike_events(
        spikes_native, native_bin_ms, cfg.bin_size_ms
    )
    num_bins, num_neurons = spike_counts.shape

    # 4. Bin cursor
    cursor_binned = _extract_cursor_position(
        cursor_native, native_bin_ms, cfg.bin_size_ms, num_bins
    )

    # 5. Velocity
    velocity = _compute_velocity(cursor_binned, cfg.bin_size_ms, mode=cfg.velocity_mode)

    # 6. Split
    train_idx, val_idx, test_idx = time_contiguous_split(
        num_bins, cfg.split_fracs, boundary_gap=cfg.boundary_gap
    )

    processed: dict[str, Any] = {
        "spike_counts": spike_counts,
        "event_times": event_times,
        "event_neurons": event_neurons,
        "velocity": velocity,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "bin_size_ms": int(cfg.bin_size_ms),
        "num_neurons": int(num_neurons),
        "dataset_name": "NLB_MC_RTT",
    }

    # 7. Validate
    _validate_processed(processed)
    logger.info(
        "preprocess complete: num_bins=%d num_neurons=%d (train=%d val=%d test=%d)",
        num_bins,
        num_neurons,
        train_idx.size,
        val_idx.size,
        test_idx.size,
    )
    return processed


# -------------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------------


def save_processed(processed: dict[str, Any], out_path: Path) -> None:
    """Persist processed-data dict to .npz, preserving object arrays."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spike_counts=processed["spike_counts"],
        event_times=np.array(processed["event_times"], dtype=object),
        event_neurons=np.array(processed["event_neurons"], dtype=object),
        velocity=processed["velocity"],
        train_idx=processed["train_idx"],
        val_idx=processed["val_idx"],
        test_idx=processed["test_idx"],
        bin_size_ms=np.array(processed["bin_size_ms"]),
        num_neurons=np.array(processed["num_neurons"]),
        dataset_name=np.array(processed["dataset_name"]),
    )
    logger.info("saved processed data to %s", out_path)


def load_processed(path: Path) -> dict[str, Any]:
    """Load processed-data .npz back into the dict shape from data_interface.md."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"processed file not found: {path}")
    z = np.load(path, allow_pickle=True)
    return {
        "spike_counts": z["spike_counts"],
        "event_times": list(z["event_times"]),
        "event_neurons": list(z["event_neurons"]),
        "velocity": z["velocity"],
        "train_idx": z["train_idx"],
        "val_idx": z["val_idx"],
        "test_idx": z["test_idx"],
        "bin_size_ms": int(z["bin_size_ms"]),
        "num_neurons": int(z["num_neurons"]),
        "dataset_name": str(z["dataset_name"]),
    }
