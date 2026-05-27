"""Velocity smoothing ablation — re-preprocess and re-run with sigma in {0, 1, 2}.

Confirms the headline R^2 numbers are not artefacts of the gaussian
smoothing on the velocity target. Re-runs preprocessing for each sigma,
then runs single-bin ridge and lag-4 ridge for direct comparison.

Writes results/smoothing_ablation/summary.json with one row per
(sigma_bins, model, event_budget).
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
logger = logging.getLogger("run_smoothing_ablation")


def _aggregate(json_path: Path) -> list[dict]:
    if not json_path.is_file():
        return []
    blob = json.loads(json_path.read_text())
    by = {}
    for r in blob.get("results", []):
        key = (r["model"], float(r["event_budget"]))
        by.setdefault(key, []).append(float(r["r2_joint"]))
    out = []
    for (model, f), vals in sorted(by.items()):
        out.append({
            "model": model, "event_budget": f,
            "mean_r2_joint": float(statistics.mean(vals)),
            "std_r2_joint": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-root", type=Path, default=Path("results/smoothing_ablation"))
    p.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    p.add_argument("--sigmas-bins", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.25])
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for sigma in args.sigmas_bins:
        sigma_tag = f"s{sigma:.1f}".replace(".", "p")
        processed_path = args.processed_root / f"processed_mc_rtt_{sigma_tag}.npz"
        if not processed_path.is_file():
            data = preprocess_mc_rtt(raw_dir=args.raw_dir, bin_size_ms=50,
                                     velocity_smooth_sigma_bins=sigma)
            save_processed(data, processed_path)
            logger.info("preprocessed sigma=%g into %s", sigma, processed_path)

        for lag in (0, 4):
            tag = f"sigma{sigma_tag}_lag{lag}"
            out_dir = args.out_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([
                sys.executable, "scripts/run_ridge.py",
                "--processed-path", str(processed_path),
                "--results-csv", str(out_dir / "results.csv"),
                "--results-json", str(out_dir / "results.json"),
                "--lag-bins", str(lag),
                "--seeds", *[str(s) for s in args.seeds],
                "--event-budgets", *[f"{b:.2f}" for b in args.event_budgets],
                "--n-boot", str(args.n_boot),
            ])
            for row in _aggregate(out_dir / "results.json"):
                row["sigma_bins"] = float(sigma)
                row["lag_bins"] = int(lag)
                summary.append(row)

    args.out_root.joinpath("summary.json").write_text(json.dumps(
        {"sigmas_bins": list(args.sigmas_bins), "rows": summary}, indent=2
    ))
    print()
    print(f"{'sigma':>6} {'lag':>4} {'model':>14} {'budget':>7} {'r2_joint':>10}")
    for r in summary:
        print(f"{r['sigma_bins']:>6.1f} {r['lag_bins']:>4} {r['model']:>14} "
              f"{r['event_budget']:>7.2f} {r['mean_r2_joint']:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
