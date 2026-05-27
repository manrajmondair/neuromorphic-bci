"""Hidden-dim scaling curve for the trained SNN.

Sweep hidden_dim in {32, 64, 128, 256, 512} at fixed k_history=4 and
f=1.0 across multiple seeds, then write a single CSV/JSON the plot
script can read. Used to identify the saturation point of the encoder
width.
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

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_hidden_dim_sweep")


def _aggregate(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    by = {}
    for r in blob["results"]:
        by.setdefault(float(r["event_budget"]), []).append(float(r["r2_joint"]))
    return [
        {
            "event_budget": f,
            "mean_r2_joint": float(statistics.mean(vs)),
            "std_r2_joint": float(statistics.stdev(vs)) if len(vs) > 1 else 0.0,
            "n_seeds": len(vs),
        }
        for f, vs in sorted(by.items(), reverse=True)
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.25])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--out-dir", type=Path, default=Path("results/hidden_dim_sweep"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for h in args.hidden_dims:
        out_path = args.out_dir / f"h{h}_results.json"
        csv_path = args.out_dir / f"h{h}_results.csv"
        logger.info("=" * 72)
        logger.info("hidden_dim=%d  k_history=%d  epochs=%d", h, args.k_history, args.epochs)
        logger.info("=" * 72)
        subprocess.check_call([
            sys.executable, "scripts/run_trained_snn.py",
            "--hidden-dim", str(h),
            "--k-history", str(args.k_history),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--seeds", *[str(s) for s in args.seeds],
            "--event-budgets", *[f"{b:.2f}" for b in args.event_budgets],
            "--n-boot", str(args.n_boot),
            "--model-name", f"trained_snn_h{args.k_history}_w{h}",
            "--results-csv", str(csv_path),
            "--results-json", str(out_path),
        ])
        for row in _aggregate(out_path):
            row["hidden_dim"] = int(h)
            row["k_history"] = int(args.k_history)
            rows.append(row)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({"rows": rows}, indent=2))
    print()
    print(f"{'hidden':>7} {'budget':>7} {'mean_r2':>10} {'std':>8}")
    for r in rows:
        print(f"{r['hidden_dim']:>7} {r['event_budget']:>7.2f} {r['mean_r2_joint']:>+10.4f} {r['std_r2_joint']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
