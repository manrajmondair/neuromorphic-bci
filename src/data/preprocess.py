"""Build the processed-data object defined in docs/data_interface.md.

Owned by data-ridge-baseline. Anything in here is contract-bearing — every
downstream model trusts the invariants listed in the data interface doc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def preprocess_mc_rtt(
    raw_path: Path,
    bin_size_ms: int = 50,
    split_fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, Any]:
    """Run full preprocessing and return the processed-data dict.

    Returns a dict with the keys specified in docs/data_interface.md.
    """
    raise NotImplementedError


def save_processed(processed: dict[str, Any], out_path: Path) -> None:
    """Persist processed-data dict to .npz, preserving object arrays."""
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


def load_processed(path: Path) -> dict[str, Any]:
    """Load processed-data .npz back into the dict shape from data_interface.md."""
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
