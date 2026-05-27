"""Permutation test for the SNN-vs-shuffle gap.

For each event budget we already have:
  * snn_r2(real-order events)
  * shuffle_r2(events with within-bin order permuted)

This script generates a null distribution of "what would the SNN R^2
look like if order didn't matter" by repeatedly re-permuting the
within-bin order with different shuffle seeds, re-fitting the SNN, and
recording the test R^2. The fraction of null R^2s that exceed the
real-order R^2 gives a one-sided p-value per budget.

Writes results/snn/permutation_test.json with one entry per budget:
  {
    "event_budget": float,
    "real_r2_joint": float,            # SNN on real-order events
    "null_r2_joints": list[float],     # `n_perm` re-permuted runs
    "null_mean": float,
    "null_std": float,
    "p_value": float,                  # P(null >= real)
    "snn_minus_null_mean": float,
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.controls.order_shuffle import shuffle_within_bin_order
from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.models.snn_decoder import SparseLatencySNN, tune_threshold_on_val
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_permutation_test")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out", type=Path, default=Path("results/snn/permutation_test.json"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.50, 0.25, 0.10])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-perm", type=int, default=100,
                   help="number of shuffle permutations per budget (null distribution size)")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument("--n-restarts", type=int, default=1)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_rows = []

    for f in args.event_budgets:
        logger.info("=" * 72)
        logger.info("permutation test: f=%.2f, n_perm=%d", f, args.n_perm)
        logger.info("=" * 72)
        et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)

        # Pick a threshold once on the val split using the real-order events.
        best_thr, _ = tune_threshold_on_val(
            num_neurons=num_neurons, bin_size_ms=bin_size_ms,
            event_times=et, event_neurons=en,
            velocity=y, train_idx=train_idx, val_idx=val_idx,
            hidden_dim=args.hidden_dim, tau_ms=args.tau_ms,
            n_restarts=1, seed=args.seed,
        )
        logger.info("budget f=%.2f tuned threshold=%g", f, best_thr)

        def _fit_predict(et_local, en_local, restart_seed):
            snn = SparseLatencySNN(
                num_neurons=num_neurons, hidden_dim=args.hidden_dim,
                tau_ms=args.tau_ms, threshold=best_thr,
                bin_size_ms=bin_size_ms,
                n_restarts=args.n_restarts, standardize=True,
                seed=restart_seed,
            ).fit(et_local, en_local, y, train_idx, val_idx)
            return snn.predict(et_local, en_local, test_idx)

        # Real-order SNN R^2 (single fit; same seed as the canonical run_snn).
        y_real = _fit_predict(et, en, args.seed)
        real_r2 = velocity_r2(y[test_idx], y_real)["r2_joint"]
        logger.info("real-order r2_joint=%+.4f", real_r2)

        # Null distribution: re-shuffle with fresh seeds.
        null_seeds = rng.integers(10_000, 10_000_000, size=args.n_perm).tolist()
        null_r2s = []
        for i, sd in enumerate(null_seeds):
            et_s, en_s = shuffle_within_bin_order(et, en, seed=int(sd))
            y_null = _fit_predict(et_s, en_s, args.seed)  # keep restart seed fixed
            null_r2 = velocity_r2(y[test_idx], y_null)["r2_joint"]
            null_r2s.append(float(null_r2))
            if (i + 1) % 20 == 0:
                logger.info("  perm %d/%d: null_r2=%+.4f (running mean=%+.4f)",
                            i + 1, args.n_perm, null_r2, float(np.mean(null_r2s)))

        nr = np.array(null_r2s, dtype=np.float64)
        # One-sided p-value: how often did the null match or beat the real fit?
        p_value = float((nr >= real_r2).sum() + 1) / float(args.n_perm + 1)
        row = {
            "event_budget": float(f),
            "tuned_threshold": float(best_thr),
            "real_r2_joint": float(real_r2),
            "null_r2_joints": null_r2s,
            "null_mean": float(nr.mean()),
            "null_std": float(nr.std(ddof=1)) if args.n_perm > 1 else 0.0,
            "null_lo_ci": float(np.percentile(nr, 2.5)),
            "null_hi_ci": float(np.percentile(nr, 97.5)),
            "p_value_one_sided": p_value,
            "snn_minus_null_mean": float(real_r2 - nr.mean()),
            "n_perm": int(args.n_perm),
            "seed": int(args.seed),
        }
        out_rows.append(row)
        logger.info(
            "budget f=%.2f  real=%+.4f  null_mean=%+.4f [%+.4f, %+.4f]  p=%.4f  gap=%+.4f",
            f, real_r2, nr.mean(),
            row["null_lo_ci"], row["null_hi_ci"], p_value, real_r2 - nr.mean(),
        )

    args.out.write_text(json.dumps({"rows": out_rows}, indent=2))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
