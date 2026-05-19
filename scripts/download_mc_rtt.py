"""Download the NLB MC_RTT dataset from DANDI into data/raw/.

Single source of truth for dataset identifiers — if the dandiset id or
version pin needs to change, change it here only. Both branches depend on
this script for reproducible dataset access.

Usage:
    python scripts/download_mc_rtt.py
    python scripts/download_mc_rtt.py --version draft
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# NLB MC_RTT dandiset. See docs/dataset.md for provenance and citation.
DANDISET_ID = "000129"
DANDISET_VERSION = "draft"
RAW_DIR = Path("data/raw")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NLB MC_RTT from DANDI.")
    parser.add_argument(
        "--version",
        default=DANDISET_VERSION,
        help=f"dandiset version to pull (default: {DANDISET_VERSION})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"directory to download into (default: {RAW_DIR})",
    )
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    dandi_url = f"DANDI:{DANDISET_ID}/{args.version}"
    print(f"downloading {dandi_url} into {args.raw_dir} ...")
    try:
        subprocess.run(
            ["dandi", "download", "-o", str(args.raw_dir), dandi_url],
            check=True,
        )
    except FileNotFoundError:
        print("error: `dandi` CLI not found. install with `pip install dandi`.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"dandi download failed (exit {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)

    print(f"done. files written under {args.raw_dir / DANDISET_ID}/")
    print("next step: python scripts/preprocess_mc_rtt.py")


if __name__ == "__main__":
    main()
