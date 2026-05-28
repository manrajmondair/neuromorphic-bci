"""Trained-SNN ensemble: average predictions across N independently-trained seeds.

Each seed trains a fresh TrainedLatencySNN with the same hyperparameters but
a different RNG. We collect their per-bin velocity predictions on the test
split and average them. The ensemble's R^2 typically lands above any single
seed because uncorrelated BPTT optimization noise cancels in the mean — a
classical bagging effect.

Writes:
  * results/cluster/trained_snn_ensemble/individual.csv   — per-seed R^2 rows
  * results/cluster/trained_snn_ensemble/ensemble.json    — ensemble R^2 + CI
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import apply_event_budget
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_trained_snn_ensemble")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/cluster/trained_snn_ensemble"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.50, 0.25, 0.10])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])

    csv_rows: list[dict] = []
    ensemble_rows: list[dict] = []

    for f in args.event_budgets:
        logger.info("=" * 72)
        logger.info("ensemble: event_budget=%.2f, seeds=%s", f, args.seeds)
        logger.info("=" * 72)
        et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)

        seed_preds: list[np.ndarray] = []
        for seed in args.seeds:
            set_global_seed(seed)
            snn = TrainedLatencySNN(
                num_neurons=num_neurons, hidden_dim=args.hidden_dim,
                tau_ms=args.tau_ms, threshold=args.threshold,
                bin_size_ms=bin_size_ms, k_history=args.k_history,
                epochs=args.epochs, patience=args.patience, seed=seed,
            ).fit(et, en, y, train_idx, val_idx)
            split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
            y_pred = snn.predict(et, en, test_idx, split_starts=split_starts)
            seed_preds.append(y_pred.astype(np.float32))

            r2 = velocity_r2(y[test_idx], y_pred)
            logger.info(
                "  seed=%d  r2_joint=%+.4f (val=%+.4f, epochs=%d)",
                seed, r2["r2_joint"], snn.best_val_r2, len(snn.history),
            )
            csv_rows.append({
                "event_budget": float(f), "seed": int(seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "best_val_r2": float(snn.best_val_r2),
                "epochs_trained": int(len(snn.history)),
            })

        stack = np.stack(seed_preds, axis=0)  # [n_seeds, n_test, 2]
        ensemble_pred = stack.mean(axis=0)
        r2_ens = velocity_r2(y[test_idx], ensemble_pred)
        r2_ens_boot = velocity_r2_bootstrap(y[test_idx], ensemble_pred, n_boot=args.n_boot, seed=0)

        individual_r2s = [row["r2_joint"] for row in csv_rows if row["event_budget"] == f]
        logger.info(
            "ENSEMBLE f=%.2f r2_joint=%+.4f [%.4f, %.4f]  (best individual=%+.4f, mean individual=%+.4f, gap=%+.4f)",
            f, r2_ens["r2_joint"], r2_ens_boot["r2_joint_ci_lo"], r2_ens_boot["r2_joint_ci_hi"],
            max(individual_r2s), float(np.mean(individual_r2s)),
            r2_ens["r2_joint"] - float(np.mean(individual_r2s)),
        )

        ensemble_rows.append({
            "model": "trained_snn_ensemble",
            "event_budget": float(f),
            "n_seeds": len(args.seeds),
            "seeds": list(args.seeds),
            "ensemble_r2_vx": r2_ens["r2_vx"],
            "ensemble_r2_vy": r2_ens["r2_vy"],
            "ensemble_r2_joint": r2_ens["r2_joint"],
            "ensemble_r2_joint_ci_lo": r2_ens_boot["r2_joint_ci_lo"],
            "ensemble_r2_joint_ci_hi": r2_ens_boot["r2_joint_ci_hi"],
            "individual_r2_joint_mean": float(np.mean(individual_r2s)),
            "individual_r2_joint_std": float(np.std(individual_r2s, ddof=1)) if len(individual_r2s) > 1 else 0.0,
            "individual_r2_joint_best": float(max(individual_r2s)),
            "ensemble_gain": float(r2_ens["r2_joint"] - np.mean(individual_r2s)),
            "n_boot": int(args.n_boot),
        })

        # Save ensemble predictions for later figure regeneration.
        np.savez(
            args.out_dir / f"ensemble_predictions_f{f:.2f}.npz",
            y_true=y[test_idx],
            y_pred=ensemble_pred.astype(np.float32),
            individual=stack,
            test_idx=test_idx,
            bin_size_ms=np.array(int(bin_size_ms)),
            event_budget=np.array(float(f)),
            n_seeds=np.array(int(len(args.seeds))),
        )

    with (args.out_dir / "individual.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    (args.out_dir / "ensemble.json").write_text(
        json.dumps({
            "model": "trained_snn_ensemble",
            "dataset": "NLB_MC_RTT",
            "metric": "velocity_r2",
            "config": {
                "hidden_dim": int(args.hidden_dim),
                "tau_ms": float(args.tau_ms),
                "threshold": float(args.threshold),
                "k_history": int(args.k_history),
                "epochs_max": int(args.epochs),
                "patience": int(args.patience),
                "n_boot": int(args.n_boot),
                "seeds": list(args.seeds),
                "n_seeds": len(args.seeds),
                "event_budgets": list(args.event_budgets),
            },
            "results": ensemble_rows,
        }, indent=2)
    )
    logger.info("wrote %s and %s", args.out_dir / "individual.csv", args.out_dir / "ensemble.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
