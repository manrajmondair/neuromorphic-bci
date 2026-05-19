"""NLB MC_RTT loading from DANDI / NWB. Owned by data-ridge-baseline branch."""
from __future__ import annotations

from pathlib import Path


def download_mc_rtt(raw_dir: Path) -> Path:
    """Download MC_RTT NWB file into raw_dir and return path to the file."""
    raise NotImplementedError


def load_mc_rtt_nwb(nwb_path: Path):
    """Open the NWB file and return (spike_times_per_neuron, cursor_pos, t_pos)."""
    raise NotImplementedError
