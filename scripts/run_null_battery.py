"""Multi-shuffle null battery.

Builds a null distribution for SNN R^2 under each of four shuffle types:

  * `order_shuffle`     within-bin (time, neuron) pair permutation
                        (the original shuffle control).
  * `phase_random`      replace within-bin spike times with uniform draws;
                        keep neuron identities and per-bin counts.
  * `neuron_shuffle`    permute the neuron-id axis dataset-wide;
                        preserves population timing, kills neuron tuning.
  * `circular_shift`    roll the entire per-bin event list by a random
                        offset; preserves every per-bin distribution
                        but decorrelates bins from velocity.

For each shuffle type at f=1.0 we re-fit the reservoir SNN (with the
per-budget threshold tuning already in run_snn.py) `n_perm` times with
fresh shuffle seeds and record test R^2. Each null distribution comes
out with mean, std, 95% CI, and a one-sided p-value vs. the real-order
fit. The four nulls together isolate which structure carries signal.

Writes results/snn/null_battery.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.controls.null_controls import (
    circular_shift_bins,
    phase_randomize_times,
    shuffle_neuron_identities,
)
from src.controls.order_shuffle import shuffle_within_bin_order
from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.models.snn_decoder import SparseLatencySNN, tune_threshold_on_val
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_null_battery")


def _fit_predict(
    et, en, *, num_neurons, bin_size_ms, hidden_dim, threshold,
    velocity, train_idx, val_idx, test_idx, fit_seed,
) -> float:
    snn = SparseLatencySNN(
        num_neurons=num_neurons, hidden_dim=hidden_dim,
        threshold=threshold, bin_size_ms=bin_size_ms,
        n_restarts=1, standardize=True, seed=fit_seed,
    ).fit(et, en, velocity, train_idx, val_idx)
    y_pred = snn.predict(et, en, test_idx)
    return float(velocity_r2(velocity[test_idx], y_pred)["r2_joint"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out", type=Path, default=Path("results/snn/null_battery.json"))
    p.add_argument("--event-budget", type=float, default=1.00,
                   help="single event budget; the null battery is most informative at the densest budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-perm", type=int, default=30,
                   help="permutations per shuffle type")
    p.add_argument("--hidden-dim", type=int, default=128)
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
    f = args.event_budget
    et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
    best_thr, _ = tune_threshold_on_val(
        num_neurons=num_neurons, bin_size_ms=bin_size_ms,
        event_times=et, event_neurons=en,
        velocity=y, train_idx=train_idx, val_idx=val_idx,
        hidden_dim=args.hidden_dim, n_restarts=1, seed=args.seed,
    )
    logger.info("budget f=%.2f tuned threshold=%g", f, best_thr)

    real_r2 = _fit_predict(
        et, en,
        num_neurons=num_neurons, bin_size_ms=bin_size_ms,
        hidden_dim=args.hidden_dim, threshold=best_thr,
        velocity=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        fit_seed=args.seed,
    )
    logger.info("real-order r2_joint=%+.4f", real_r2)

    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(10_000, 10_000_000, size=args.n_perm).tolist()

    shuffle_kinds: dict[str, callable] = {
        "order_shuffle": lambda sd: shuffle_within_bin_order(et, en, seed=int(sd)),
        "phase_random": lambda sd: phase_randomize_times(et, en, bin_size_ms=bin_size_ms, seed=int(sd)),
        "neuron_shuffle": lambda sd: shuffle_neuron_identities(et, en, num_neurons=num_neurons, seed=int(sd)),
        "circular_shift": lambda sd: circular_shift_bins(et, en, seed=int(sd)),
    }

    out_rows: list[dict] = []
    for name, transform in shuffle_kinds.items():
        logger.info("=" * 72)
        logger.info("null: %s, n_perm=%d", name, args.n_perm)
        logger.info("=" * 72)
        null_r2s: list[float] = []
        for i, sd in enumerate(seeds):
            et_n, en_n = transform(sd)
            r2 = _fit_predict(
                et_n, en_n,
                num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                hidden_dim=args.hidden_dim, threshold=best_thr,
                velocity=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                fit_seed=args.seed,
            )
            null_r2s.append(r2)
            if (i + 1) % 10 == 0:
                logger.info("  perm %d/%d: null=%+.4f running_mean=%+.4f",
                            i + 1, args.n_perm, r2, float(np.mean(null_r2s)))
        nr = np.array(null_r2s, dtype=np.float64)
        p_value = float((nr >= real_r2).sum() + 1) / float(args.n_perm + 1)
        row = {
            "shuffle": name,
            "event_budget": float(f),
            "real_r2_joint": float(real_r2),
            "tuned_threshold": float(best_thr),
            "null_r2_joints": null_r2s,
            "null_mean": float(nr.mean()),
            "null_std": float(nr.std(ddof=1)) if args.n_perm > 1 else 0.0,
            "null_lo_ci": float(np.percentile(nr, 2.5)),
            "null_hi_ci": float(np.percentile(nr, 97.5)),
            "p_value_one_sided": p_value,
            "real_minus_null_mean": float(real_r2 - nr.mean()),
            "n_perm": int(args.n_perm),
            "seed": int(args.seed),
        }
        out_rows.append(row)
        logger.info(
            "%s: real=%+.4f null_mean=%+.4f [%+.4f, %+.4f] p=%.4f gap=%+.4f",
            name, real_r2, nr.mean(),
            row["null_lo_ci"], row["null_hi_ci"], p_value, row["real_minus_null_mean"],
        )

    args.out.write_text(json.dumps({"rows": out_rows}, indent=2))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
