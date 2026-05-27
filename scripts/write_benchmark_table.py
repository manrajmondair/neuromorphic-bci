"""Emit the published-baseline table as a JSON file the plotter can read."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark_table import benchmark_table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=Path("results/benchmark/nlb_mc_rtt_published.json"),
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(benchmark_table(), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
