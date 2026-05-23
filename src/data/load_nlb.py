"""NLB MC_RTT loading from DANDI / NWB.

Single source of truth for how we locate and open the dandiset on disk and
how we wrap it with `nlb_tools` for downstream consumption.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# See docs/dataset.md for full provenance and citation.
DANDISET_ID = "000129"


def find_nwb_file(raw_dir: Path) -> Path:
    """Locate the canonical MC_RTT NWB file under `raw_dir`.

    Prefers the dandi-downloaded layout `raw_dir/000129/**/*.nwb`. Falls
    back to a broader recursive search if the dandiset directory isn't
    present. If multiple candidates exist, prefers ones containing "train"
    in the filename (the file with visible behavior labels).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    dandiset_dir = raw_dir / DANDISET_ID
    if dandiset_dir.is_dir():
        candidates = sorted(dandiset_dir.rglob("*.nwb"))
        search_root = dandiset_dir
    else:
        candidates = sorted(raw_dir.rglob("*.nwb"))
        search_root = raw_dir

    if not candidates:
        raise FileNotFoundError(
            f"no .nwb file found under {search_root}. "
            f"run `python scripts/download_mc_rtt.py` first."
        )

    train_files = [p for p in candidates if "train" in p.name.lower()]
    chosen = train_files[0] if train_files else candidates[0]
    logger.info("located NWB file: %s", chosen)
    if len(candidates) > 1:
        logger.info(
            "found %d candidate NWB files under %s; chose %s",
            len(candidates),
            search_root,
            chosen.name,
        )
    return chosen


def load_mc_rtt_dataset(nwb_path: Path) -> Any:
    """Open the NWB file via `nlb_tools` at its native bin width.

    Returns an `NWBDataset` whose `.data` is a pandas DataFrame with
    MultiIndex columns `(signal, channel)` and `.bin_width` is the native
    resolution in ms (typically 1 ms for MC_RTT).
    """
    try:
        from nlb_tools.nwb_interface import NWBDataset
    except ImportError as e:
        raise ImportError(
            "nlb_tools not installed. run `pip install -r requirements.txt`."
        ) from e

    nwb_path = Path(nwb_path)
    if not nwb_path.is_file():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    logger.info("opening NWBDataset from %s", nwb_path)
    dataset = NWBDataset(fpath=str(nwb_path))
    native_bin = getattr(dataset, "bin_width", None)
    logger.info("native bin width: %s ms", native_bin)
    n_samples = len(dataset.data) if hasattr(dataset, "data") else None
    if n_samples is not None and native_bin is not None:
        logger.info(
            "recording length: %d native samples (~%.2f s)",
            n_samples,
            n_samples * native_bin / 1000.0,
        )
    return dataset
