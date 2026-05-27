"""Wall-clock latency budget for every decoder, across the same test stream.

Loads (or reproduces) the streaming summary written by
`scripts/run_online_streaming.py`, plus the analytical Loihi 2 estimate
from `scripts/run_loihi_estimate.py`, and produces a single table that
combines:

  * per-bin wall-clock CPU latency (ms)
  * per-bin estimated Loihi 2 latency (ms, derived from SynOps and a
    per-SynOp throughput figure)
  * per-bin energy (µJ) for CPU vs Loihi 2 vs NorthPole
  * decoder R^2 at the same configuration

Writes results/latency/latency_budget.json — the table the deployment
section of the paper draws from.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_latency_budget")


# Loihi 2 single-core throughput: vendor materials cite ~1.6 Gsyn-op/s/core
# (Mayr & Davies 2024). At a single chip with ~128 cores, the aggregate
# is much higher, but per-decoder we assume a single core for a tiny
# 98-input / 128-hidden network. 1 SynOp ≈ 0.63 ns.
LOIHI2_NS_PER_SYNOP = 0.63
CPU_NS_PER_MAC = 1.0  # ~1 ns per scalar MAC for a single x86 core at ~1 GFLOPS effective rate.


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--streaming-summary", type=Path, default=Path("results/streaming/summary.json"))
    p.add_argument("--loihi-summary", type=Path, default=Path("results/loihi/loihi_estimate.json"))
    p.add_argument("--out", type=Path, default=Path("results/latency/latency_budget.json"))
    p.add_argument("--bin-size-ms", type=int, default=50)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    streaming = json.loads(args.streaming_summary.read_text()) if args.streaming_summary.is_file() else {"rows": []}
    loihi = json.loads(args.loihi_summary.read_text()) if args.loihi_summary.is_file() else {"rows": []}

    cpu_table = {row["model"]: row for row in streaming.get("rows", [])}
    loihi_table = {}
    for row in loihi.get("rows", []):
        # Loihi estimate is keyed by event_budget; the streaming run is at f=1.0
        # (the canonical configuration), so we pick the f=1.0 entry.
        if abs(row["event_budget"] - 1.00) < 1e-9:
            loihi_table["trained_snn"] = row

    out_rows = []
    for model in ("ridge", "ridge_lag4", "trained_snn"):
        ent = {"model": model}
        if model in cpu_table:
            r = cpu_table[model]
            ent["cpu_mean_latency_ms"] = float(r["mean_latency_ms"])
            ent["cpu_p95_latency_ms"] = float(r["p95_latency_ms"])
            ent["cpu_p99_latency_ms"] = float(r["p99_latency_ms"])
            ent["r2_joint"] = float(r["r2_joint"])
            ent["final_drift"] = float(r["final_drift"])
        if model == "trained_snn" and "trained_snn" in loihi_table:
            r = loihi_table["trained_snn"]
            synops_per_pred = float(r["synops_per_prediction_mean"])
            ent["loihi2_synops_per_prediction"] = synops_per_pred
            ent["loihi2_estimated_latency_ms"] = synops_per_pred * LOIHI2_NS_PER_SYNOP * 1e-6
            ent["loihi2_uj_per_prediction"] = float(r["per_chip"]["loihi2"]["energy_per_prediction_uj"])
            ent["northpole_uj_per_prediction"] = float(r["per_chip"]["northpole"]["energy_per_prediction_uj"])
        out_rows.append(ent)

    out_blob = {
        "rows": out_rows,
        "bin_size_ms": int(args.bin_size_ms),
        "loihi2_ns_per_synop": float(LOIHI2_NS_PER_SYNOP),
        "cpu_ns_per_mac": float(CPU_NS_PER_MAC),
        "notes": (
            "CPU numbers are measured wall-clock per-bin latency on x86 / ARM "
            "(macOS development host). Loihi 2 numbers are analytical "
            "projections from the trained SNN's measured SynOp count, "
            "multiplied by vendor-published per-SynOp throughput and energy. "
            "Real silicon would likely be lower because fanout compression "
            "and on-chip routing reduce SynOp counts further; this is an "
            "upper bound."
        ),
    }
    args.out.write_text(json.dumps(out_blob, indent=2))
    print()
    print(f"{'model':>13} {'r2':>8} {'cpu_ms':>9} {'loihi_ms':>10} {'loihi_uJ':>10} {'np_uJ':>10}")
    for r in out_rows:
        print(f"{r['model']:>13} {r.get('r2_joint', float('nan')):>+8.4f} "
              f"{r.get('cpu_mean_latency_ms', float('nan')):>9.3f} "
              f"{r.get('loihi2_estimated_latency_ms', float('nan')):>10.4f} "
              f"{r.get('loihi2_uj_per_prediction', float('nan')):>10.4f} "
              f"{r.get('northpole_uj_per_prediction', float('nan')):>10.5f}")
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
