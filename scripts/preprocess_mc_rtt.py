"""Run NLB MC_RTT preprocessing and write data/processed/mc_rtt.npz.

Owned by data-ridge-baseline.
"""
from __future__ import annotations

from pathlib import Path

from src.data.preprocess import preprocess_mc_rtt, save_processed


def main() -> None:
    raw_dir = Path("data/raw")
    out_path = Path("data/processed/mc_rtt.npz")
    processed = preprocess_mc_rtt(raw_dir, bin_size_ms=50)
    save_processed(processed, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
