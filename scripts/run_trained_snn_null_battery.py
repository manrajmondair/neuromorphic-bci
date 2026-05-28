"""Multi-shuffle null battery for the **trained** SNN.

Mirrors `run_null_battery.py` (which uses the reservoir SNN) but
substitutes `TrainedLatencySNN` so the headline decoder is the one whose
null distribution we report. Same four shuffle controls:

  * `order_shuffle`    within-bin permutation of the (time, neuron) pairs.
  * `phase_random`     replace within-bin times with uniform draws;
                       neuron identities and per-bin counts preserved.
  * `neuron_shuffle`   dataset-wide permutation of the neuron-id axis;
                       preserves population timing, kills neuron tuning.
  * `circular_shift`   random roll of the entire per-bin event list;
                       preserves every per-bin distribution but
                       decorrelates bins from velocity.

Each (event_budget, shuffle) cell trains `n_perm` independent
trained_snn fits on freshly shuffled inputs and records test R^2,
giving a null distribution with mean ± std, 95% CI, and a one-sided
p-value vs. the real-order fit.

Writes results/cluster/snn_trained/null_battery.json. The schema
matches `results/snn/null_battery.json` so the same downstream tooling
works.
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
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_trained_snn_null_battery")


def _fit_predict(
    et,
    en,
    *,
    num_neurons,
    bin_size_ms,
    hidden_dim,
    tau_ms,
    threshold,
    k_history,
    epochs,
    patience,
    velocity,
    train_idx,
    val_idx,
    test_idx,
    fit_seed,
) -> float:
    snn = TrainedLatencySNN(
        num_neurons=num_neurons,
        hidden_dim=hidden_dim,
        tau_ms=tau_ms,
        threshold=threshold,
        bin_size_ms=bin_size_ms,
        k_history=k_history,
        epochs=epochs,
        patience=patience,
        seed=fit_seed,
    ).fit(et, en, velocity, train_idx, val_idx)
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    y_pred = snn.predict(et, en, test_idx, split_starts=split_starts)
    return float(velocity_r2(velocity[test_idx], y_pred)["r2_joint"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz")
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/cluster/snn_trained/null_battery.json"),
    )
    p.add_argument(
        "--event-budgets",
        type=float,
        nargs="+",
        default=[1.00, 0.50, 0.25, 0.10],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-perm",
        type=int,
        default=20,
        help="permutations per (shuffle, budget) cell (null distribution size)",
    )
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.30,
                   help="used when --tune-threshold is OFF; otherwise a starting point only")
    p.add_argument(
        "--tune-threshold",
        action="store_true",
        help="tune the trained-SNN threshold on the val split per event budget "
             "before running the null loop. Mirrors run_null_battery.py's per-cell "
             "tune_threshold_on_val for the reservoir SNN — costs a small handful "
             "of extra fits per budget but makes the trained- and reservoir-null "
             "batteries methodologically comparable.",
    )
    p.add_argument(
        "--tune-threshold-grid",
        type=float,
        nargs="+",
        default=[0.15, 0.30, 0.50, 0.80],
        help="threshold values searched when --tune-threshold is set",
    )
    p.add_argument(
        "--vary-fit-seed",
        action="store_true",
        help="use a different model-init seed for every null permutation. "
             "Default (off) preserves model init across perms so the null "
             "isolates data-shuffle variance; --vary-fit-seed widens the null "
             "to also sample fitting variance (more conservative p-values).",
    )
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
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

    out_rows: list[dict] = []
    for f in args.event_budgets:
        logger.info("#" * 72)
        logger.info("event_budget=%.2f", f)
        logger.info("#" * 72)
        et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)

        # Per-budget threshold tuning. Mirrors run_null_battery.py's
        # tune_threshold_on_val pattern for the reservoir SNN so the two
        # null batteries are methodologically comparable.
        if args.tune_threshold:
            best_thr = float(args.threshold)
            best_val = float("-inf")
            for thr in args.tune_threshold_grid:
                snn_tune = TrainedLatencySNN(
                    num_neurons=num_neurons,
                    hidden_dim=args.hidden_dim,
                    tau_ms=args.tau_ms,
                    threshold=float(thr),
                    bin_size_ms=bin_size_ms,
                    k_history=args.k_history,
                    epochs=args.epochs,
                    patience=args.patience,
                    seed=args.seed,
                ).fit(et, en, y, train_idx, val_idx)
                if snn_tune.best_val_r2 > best_val:
                    best_val = float(snn_tune.best_val_r2)
                    best_thr = float(thr)
                logger.info("tune f=%.2f thr=%g  val_r2=%+.4f", f, thr, snn_tune.best_val_r2)
            cell_threshold = best_thr
            logger.info("tuned threshold for f=%.2f: %g (val_r2=%+.4f)", f, cell_threshold, best_val)
        else:
            cell_threshold = float(args.threshold)

        logger.info("fitting real-order trained_snn ...")
        real_r2 = _fit_predict(
            et,
            en,
            num_neurons=num_neurons,
            bin_size_ms=bin_size_ms,
            hidden_dim=args.hidden_dim,
            tau_ms=args.tau_ms,
            threshold=cell_threshold,
            k_history=args.k_history,
            epochs=args.epochs,
            patience=args.patience,
            velocity=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            fit_seed=args.seed,
        )
        logger.info("real-order r2_joint=%+.4f", real_r2)

        rng = np.random.default_rng(args.seed + int(f * 100))
        seeds = rng.integers(10_000, 10_000_000, size=args.n_perm).tolist()

        shuffle_kinds = {
            "order_shuffle": lambda sd: shuffle_within_bin_order(et, en, seed=int(sd)),
            "phase_random": lambda sd: phase_randomize_times(
                et, en, bin_size_ms=bin_size_ms, seed=int(sd)
            ),
            "neuron_shuffle": lambda sd: shuffle_neuron_identities(
                et, en, num_neurons=num_neurons, seed=int(sd)
            ),
            "circular_shift": lambda sd: circular_shift_bins(et, en, seed=int(sd)),
        }

        for name, transform in shuffle_kinds.items():
            logger.info("-" * 72)
            logger.info("null=%s f=%.2f n_perm=%d", name, f, args.n_perm)
            logger.info("-" * 72)
            null_r2s: list[float] = []
            for i, sd in enumerate(seeds):
                et_n, en_n = transform(sd)
                # --vary-fit-seed widens the null by varying model init per
                # perm (samples both data-shuffle AND fit variance, more
                # conservative); default keeps fit_seed=args.seed to isolate
                # data-shuffle variance only.
                fit_seed_this = int(sd) if args.vary_fit_seed else args.seed
                r2 = _fit_predict(
                    et_n,
                    en_n,
                    num_neurons=num_neurons,
                    bin_size_ms=bin_size_ms,
                    hidden_dim=args.hidden_dim,
                    tau_ms=args.tau_ms,
                    threshold=cell_threshold,
                    k_history=args.k_history,
                    epochs=args.epochs,
                    patience=args.patience,
                    velocity=y,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    fit_seed=fit_seed_this,
                )
                null_r2s.append(r2)
                if (i + 1) % 5 == 0:
                    logger.info(
                        "  perm %d/%d: null=%+.4f running_mean=%+.4f",
                        i + 1,
                        args.n_perm,
                        r2,
                        float(np.mean(null_r2s)),
                    )

            nr = np.array(null_r2s, dtype=np.float64)
            p_value = float((nr >= real_r2).sum() + 1) / float(args.n_perm + 1)
            row = {
                "shuffle": name,
                "event_budget": float(f),
                "real_r2_joint": float(real_r2),
                "tuned_threshold": float(cell_threshold),
                "threshold_tuned_per_budget": bool(args.tune_threshold),
                "vary_fit_seed": bool(args.vary_fit_seed),
                "null_r2_joints": null_r2s,
                "null_mean": float(nr.mean()),
                "null_std": float(nr.std(ddof=1)) if args.n_perm > 1 else 0.0,
                "null_lo_ci": float(np.percentile(nr, 2.5)),
                "null_hi_ci": float(np.percentile(nr, 97.5)),
                "p_value_one_sided": p_value,
                "real_minus_null_mean": float(real_r2 - nr.mean()),
                "n_perm": int(args.n_perm),
                "seed": int(args.seed),
                "model": "trained_snn",
                "hidden_dim": int(args.hidden_dim),
                "k_history": int(args.k_history),
                "tau_ms": float(args.tau_ms),
                "threshold": float(args.threshold),
                "epochs": int(args.epochs),
            }
            out_rows.append(row)
            logger.info(
                "%s f=%.2f: real=%+.4f null_mean=%+.4f [%+.4f, %+.4f] p=%.4f gap=%+.4f",
                name,
                f,
                real_r2,
                nr.mean(),
                row["null_lo_ci"],
                row["null_hi_ci"],
                p_value,
                row["real_minus_null_mean"],
            )

            # Stream after every cell so partial results survive a crash.
            args.out.write_text(json.dumps({"rows": out_rows}, indent=2))

    args.out.write_text(json.dumps({"rows": out_rows}, indent=2))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
