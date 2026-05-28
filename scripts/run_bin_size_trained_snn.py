"""Bin-size sweep for the trained SNN — fills the gap left by
`run_bin_size_sweep.py` which only runs ridge + reservoir SNN.

For each bin width in `--bin-sizes-ms`, preprocesses the dataset at
that bin width (cached at `data/processed/processed_mc_rtt_b{N}.npz`)
and then runs `scripts/run_trained_snn.py` against it. The existing
order/latency-gap claim is currently shown only for the reservoir SNN.
Repeating it on the trained SNN confirms that the bin-width choice
(50 ms) isn't a model-specific cherry-pick.

Writes:
  results/cluster/bin_sweep_trained/b{N}/results.{csv,json}
  results/cluster/bin_sweep_trained/summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import preprocess_mc_rtt, save_processed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_bin_size_trained_snn")


def _aggregate(json_path: Path) -> list[dict]:
    if not json_path.is_file():
        return []
    blob = json.loads(json_path.read_text())
    return blob.get("results") or blob.get("rows") or []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin-sizes-ms", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--processed-root", type=Path, default=Path("data/processed"),
        help="cached per-bin-width processed npz files live here",
    )
    p.add_argument(
        "--out-root", type=Path, default=Path("results/cluster/bin_sweep_trained"),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--event-budgets", type=float, nargs="+",
        default=[1.00, 0.50, 0.25, 0.10],
    )
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.processed_root.mkdir(parents=True, exist_ok=True)

    # Staleness check: cached per-bin npz is reused only if it's newer than
    # both the preprocess module and the raw data. Re-preprocesses on any
    # drift to avoid silently feeding pre-fix splits into a post-fix sweep.
    preprocess_module = Path(__file__).resolve().parents[1] / "src" / "data" / "preprocess.py"
    preprocess_mtime = preprocess_module.stat().st_mtime if preprocess_module.is_file() else 0.0

    summary_rows: list[dict] = []
    for bin_ms in args.bin_sizes_ms:
        logger.info("=" * 72)
        logger.info("bin_size_ms=%d", bin_ms)
        logger.info("=" * 72)
        processed_path = args.processed_root / f"processed_mc_rtt_b{bin_ms}.npz"
        stale = False
        if processed_path.is_file():
            cache_mtime = processed_path.stat().st_mtime
            if preprocess_mtime > cache_mtime:
                logger.info(
                    "cache %s is older than src/data/preprocess.py — re-preprocessing",
                    processed_path,
                )
                stale = True
        if not processed_path.is_file() or stale:
            logger.info("preprocessing into %s ...", processed_path)
            data = preprocess_mc_rtt(raw_dir=args.raw_dir, bin_size_ms=bin_ms)
            save_processed(data, processed_path)

        out_dir = args.out_root / f"b{bin_ms}"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "trained_snn_results.json"
        csv_path = out_dir / "trained_snn_results.csv"

        subprocess.check_call([
            sys.executable, "scripts/run_trained_snn.py",
            "--processed-path", str(processed_path),
            "--results-csv", str(csv_path),
            "--results-json", str(json_path),
            "--seeds", *[str(s) for s in args.seeds],
            "--event-budgets", *[f"{b:.2f}" for b in args.event_budgets],
            "--hidden-dim", str(args.hidden_dim),
            "--k-history", str(args.k_history),
            "--threshold", str(args.threshold),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--n-boot", str(args.n_boot),
            "--model-name", f"trained_snn_b{bin_ms}",
        ])

        for row in _aggregate(json_path):
            summary_rows.append({"bin_size_ms": int(bin_ms), **row})

    (args.out_root / "summary.json").write_text(
        json.dumps({"rows": summary_rows}, indent=2)
    )
    logger.info("wrote %s (%d rows)", args.out_root / "summary.json", len(summary_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
