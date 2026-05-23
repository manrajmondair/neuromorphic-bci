"""Run NLB MC_RTT preprocessing and write data/processed/processed_mc_rtt.npz.

Owned by data-ridge-baseline. The post-save validation block at the bottom
of `main()` enforces the data contract: it re-loads the .npz, prints every
array's shape/dtype, asserts strict-monotonic event_times in every bin, and
asserts no NaN in velocity. CI / future-Manraj reads the validation block
as the canonical record of what shapes everything should have.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed, preprocess_mc_rtt, save_processed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("preprocess_mc_rtt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess NLB MC_RTT into processed_mc_rtt.npz")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--output", type=Path, default=Path("data/processed/processed_mc_rtt.npz")
    )
    p.add_argument("--bin-size-ms", type=int, default=50)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument(
        "--boundary-gap",
        type=int,
        default=1,
        help="bins dropped on each side of each split boundary to prevent label leakage",
    )
    p.add_argument("--velocity-mode", choices=("central", "forward"), default="central")
    p.add_argument(
        "--behavior-signal",
        default="cursor_pos",
        help="primary behavior signal; falls back to finger_pos[:, :2] if absent",
    )
    p.add_argument(
        "--velocity-smooth-sigma-bins",
        type=float,
        default=1.0,
        help="Gaussian sigma in bins applied to the velocity target (0 disables)",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def post_save_validation(npz_path: Path) -> None:
    """Re-load the saved .npz and enforce every contract invariant.

    Specifically:
      * print shapes/dtypes of every key
      * assert event_times in every bin is strictly monotonically increasing
      * assert velocity contains no NaN
      * assert splits are disjoint and in-range
    """
    logger.info("=" * 72)
    logger.info("post-save validation: re-loading %s", npz_path)
    logger.info("=" * 72)
    data = load_processed(npz_path)

    spike_counts = data["spike_counts"]
    event_times = data["event_times"]
    event_neurons = data["event_neurons"]
    velocity = data["velocity"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    num_bins = len(event_times)

    logger.info("keys present: %s", sorted(data.keys()))
    logger.info(
        "spike_counts: shape=%s dtype=%s total=%d",
        spike_counts.shape,
        spike_counts.dtype,
        int(spike_counts.sum()),
    )
    logger.info(
        "event_times:   object-array of %d bins, total events=%d, mean=%.2f/bin",
        num_bins,
        sum(t.size for t in event_times),
        sum(t.size for t in event_times) / max(num_bins, 1),
    )
    logger.info(
        "event_neurons: object-array of %d bins (1:1 with event_times)",
        len(event_neurons),
    )
    logger.info(
        "velocity:      shape=%s dtype=%s",
        velocity.shape,
        velocity.dtype,
    )
    logger.info(
        "splits: train=%s val=%s test=%s (total bins covered=%d of %d)",
        train_idx.shape,
        val_idx.shape,
        test_idx.shape,
        train_idx.size + val_idx.size + test_idx.size,
        num_bins,
    )
    logger.info(
        "metadata: bin_size_ms=%s num_neurons=%s dataset_name=%s",
        data["bin_size_ms"],
        data["num_neurons"],
        data["dataset_name"],
    )

    # strict-monotonic event_times across every bin
    bad_bins: list[int] = []
    for t in range(num_bins):
        times = event_times[t]
        if times.size > 1 and not np.all(np.diff(times) > 0):
            bad_bins.append(t)
    if bad_bins:
        raise AssertionError(
            f"event_times not strictly monotonic in {len(bad_bins)} bins "
            f"(first few: {bad_bins[:5]})"
        )
    logger.info("strict-monotonic event_times: PASS across %d bins", num_bins)

    # no NaN in velocity targets
    if np.isnan(velocity).any():
        raise AssertionError(f"velocity contains {int(np.isnan(velocity).sum())} NaN values")
    logger.info("velocity NaN check: PASS (no NaN in %s tensor)", velocity.shape)

    # splits disjoint and in-range
    union = np.concatenate([train_idx, val_idx, test_idx])
    if len(set(union.tolist())) != union.size:
        raise AssertionError("split indices overlap")
    if union.min() < 0 or union.max() >= num_bins:
        raise AssertionError("split indices out of range")
    logger.info(
        "split disjointness + range: PASS (covered=%d, gap=%d bins by design)",
        union.size,
        num_bins - union.size,
    )

    logger.info("=" * 72)
    logger.info("post-save validation: ALL CHECKS PASS")
    logger.info("=" * 72)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    fracs = (args.train_frac, args.val_frac, args.test_frac)
    processed = preprocess_mc_rtt(
        raw_dir=args.raw_dir,
        bin_size_ms=args.bin_size_ms,
        split_fracs=fracs,
        boundary_gap=args.boundary_gap,
        velocity_mode=args.velocity_mode,
        behavior_signal=args.behavior_signal,
        velocity_smooth_sigma_bins=args.velocity_smooth_sigma_bins,
    )
    save_processed(processed, args.output)
    size_mb = args.output.stat().st_size / 1e6
    logger.info("wrote %s (%.2f MB)", args.output, size_mb)

    post_save_validation(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
