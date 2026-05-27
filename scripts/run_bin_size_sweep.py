"""Bin-size sweep — replays the full ridge + SNN + shuffle pipeline at 20, 50, 100 ms.

For each bin size we:
  1. Preprocess the raw NWB into `data/processed/processed_mc_rtt_b{bin_ms}.npz`
     using `preprocess_mc_rtt` directly (re-using the existing pipeline so
     splits, gap handling, NaN cleaning, velocity smoothing, and jitter are
     identical to the 50 ms default).
  2. Run `scripts/run_ridge.py` (single-bin) and `scripts/run_snn.py`
     (reservoir SNN + shuffle control) against that npz, writing each
     model's outputs to `results/bin_sweep/b{bin_ms}/{model}_results.json`.

Writes a small aggregate summary at the end:
  * results/bin_sweep/summary.json
    one row per (model, bin_ms, event_budget) with mean r2_joint across seeds.

The proposal's claim is that smaller bins should expose more order/latency
structure, so the SNN-vs-shuffle gap should widen as bin size shrinks. This
script produces the table that test rests on.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import preprocess_mc_rtt, save_processed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_bin_size_sweep")

DEFAULT_BIN_SIZES_MS = (20, 50, 100)


def _aggregate(json_path: Path) -> list[dict]:
    if not json_path.is_file():
        return []
    blob = json.loads(json_path.read_text())
    rows = blob.get("results", [])
    by = {}
    for r in rows:
        key = (r["model"], float(r["event_budget"]))
        by.setdefault(key, []).append(float(r["r2_joint"]))
    out = []
    for (model, f), vals in sorted(by.items()):
        out.append({
            "model": model,
            "event_budget": f,
            "mean_r2_joint": float(statistics.mean(vals)),
            "std_r2_joint": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin-sizes-ms", type=int, nargs="+", default=list(DEFAULT_BIN_SIZES_MS))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-root", type=Path, default=Path("results/bin_sweep"))
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.50, 0.25, 0.10])
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.processed_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for bin_ms in args.bin_sizes_ms:
        logger.info("=" * 72)
        logger.info("bin_size_ms=%d", bin_ms)
        logger.info("=" * 72)
        processed_path = args.processed_root / f"processed_mc_rtt_b{bin_ms}.npz"
        if not processed_path.is_file():
            data = preprocess_mc_rtt(raw_dir=args.raw_dir, bin_size_ms=bin_ms)
            save_processed(data, processed_path)
        out_dir = args.out_root / f"b{bin_ms}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("ridge", "snn", "controls"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

        # Ridge (single-bin counts)
        subprocess.check_call([
            sys.executable, "scripts/run_ridge.py",
            "--processed-path", str(processed_path),
            "--results-csv", str(out_dir / "ridge" / "results.csv"),
            "--results-json", str(out_dir / "ridge" / "ridge_results.json"),
            "--seeds", *[str(s) for s in args.seeds],
            "--event-budgets", *[f"{b:.2f}" for b in args.event_budgets],
            "--n-boot", str(args.n_boot),
        ])
        # SNN + shuffle
        subprocess.check_call([
            sys.executable, "scripts/run_snn.py",
            "--processed-path", str(processed_path),
            "--snn-csv", str(out_dir / "snn" / "results.csv"),
            "--snn-json", str(out_dir / "snn" / "snn_results.json"),
            "--shuffle-csv", str(out_dir / "controls" / "results.csv"),
            "--shuffle-json", str(out_dir / "controls" / "shuffle_results.json"),
            "--seeds", *[str(s) for s in args.seeds],
            "--event-budgets", *[f"{b:.2f}" for b in args.event_budgets],
            "--n-boot", str(args.n_boot),
        ])

        for sub, fname in (
            ("ridge", "ridge_results.json"),
            ("snn", "snn_results.json"),
            ("controls", "shuffle_results.json"),
        ):
            agg = _aggregate(out_dir / sub / fname)
            for row in agg:
                row["bin_size_ms"] = int(bin_ms)
                summary.append(row)

    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(
        {"bin_sizes_ms": list(args.bin_sizes_ms), "rows": summary}, indent=2
    ))
    logger.info("wrote %s", summary_path)

    # Quick gap table to stdout
    rows_by = {}
    for r in summary:
        rows_by[(r["bin_size_ms"], r["event_budget"], r["model"])] = r["mean_r2_joint"]
    print()
    print(f"{'bin_ms':>7} {'budget':>7} {'snn':>10} {'shuffle':>10} {'gap':>10} {'ridge':>10}")
    for bin_ms in args.bin_sizes_ms:
        for f in args.event_budgets:
            snn = rows_by.get((bin_ms, f, "snn"), float("nan"))
            shuf = rows_by.get((bin_ms, f, "snn_shuffle"), float("nan"))
            ridge = rows_by.get((bin_ms, f, "ridge"), float("nan"))
            print(f"{bin_ms:>7} {f:>7.2f} {snn:>+10.4f} {shuf:>+10.4f} {snn - shuf:>+10.4f} {ridge:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
