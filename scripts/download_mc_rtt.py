"""Download the NLB MC_RTT dataset from DANDI into data/raw/.

Single source of truth for dataset identifiers — if the dandiset id or
version pin needs to change, change it here only. Both branches depend
on this script for reproducible dataset access.

Usage:
    python scripts/download_mc_rtt.py
    python scripts/download_mc_rtt.py --version draft
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# NLB MC_RTT dandiset. See docs/dataset.md for provenance and citation.
DANDISET_ID = "000129"
DANDISET_VERSION = "draft"
RAW_DIR = Path("data/raw")

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("download_mc_rtt")


def main() -> int:
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
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from dandi.download import download
    except ImportError as e:
        logger.error("dandi package not installed in this interpreter: %s", e)
        logger.error("install via `pip install -r requirements.txt`")
        return 1

    dandi_url = f"https://dandiarchive.org/dandiset/{DANDISET_ID}/{args.version}"
    logger.info("downloading %s into %s ...", dandi_url, args.raw_dir)
    download(dandi_url, output_dir=str(args.raw_dir))

    files = sorted(args.raw_dir.rglob("*.nwb"))
    if not files:
        logger.error("download finished but no .nwb files appeared under %s", args.raw_dir)
        return 1
    for f in files:
        logger.info("  %s (%.1f MB)", f, f.stat().st_size / 1e6)
    logger.info("done. next step: python scripts/preprocess_mc_rtt.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
