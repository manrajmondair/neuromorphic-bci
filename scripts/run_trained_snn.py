"""Train the end-to-end trainable SNN (BPTT + surrogate gradients) across event budgets.

For each (seed, event_budget) we fit a TrainedLatencySNN with Adam over
the train split, early-stopping on val joint R^2, then score on test
with bootstrap CIs. The threshold / tau / hidden_dim defaults are
small on purpose — this is meant to be a fair "actually trained"
counterpoint to the random-projection reservoir, not a giant model.

Writes:
  * results/trained_snn/trained_snn_results.json
  * results/trained_snn/results.csv
  * results/trained_snn/predictions_f0.25_seed0.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.experiment_runner import append_result, save_json_results
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import apply_event_budget
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_trained_snn")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)
SEEDS_DEFAULT = (0, 1, 2)
QUALITATIVE_BUDGET = 0.25
QUALITATIVE_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/trained_snn/results.csv"))
    p.add_argument("--results-json", type=Path, default=Path("results/trained_snn/trained_snn_results.json"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=list(EVENT_BUDGETS_DEFAULT))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--tau-ms", type=float, default=10.0)
    # Defaults from a small grid sweep at f=1.0: threshold 0.3 + lr 1e-2 hits
    # r2_joint > 0.18 (beats reservoir SNN + single-bin ridge) within ~80 epochs.
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--surrogate-slope", type=float, default=25.0)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])
    if args.results_csv.exists():
        args.results_csv.unlink()
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    n_events_total = int(sum(t.size for t in data["event_times"]))
    rows: list[dict] = []
    for seed in args.seeds:
        set_global_seed(seed)
        for f in args.event_budgets:
            logger.info("=" * 72)
            logger.info("trained_snn: seed=%d event_budget=%.2f", seed, f)
            logger.info("=" * 72)
            et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
            n_events_used = int(sum(t.size for t in et))

            snn = TrainedLatencySNN(
                num_neurons=num_neurons,
                hidden_dim=args.hidden_dim,
                tau_ms=args.tau_ms,
                threshold=args.threshold,
                bin_size_ms=bin_size_ms,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                surrogate_slope=args.surrogate_slope,
                seed=seed,
            ).fit(et, en, y, train_idx, val_idx)
            y_pred = snn.predict(et, en, test_idx)
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
            logger.info(
                "trained_snn result: f=%.2f seed=%d  r2_joint=%+.4f [%.4f, %.4f]  best_val_r2=%+.4f  epochs=%d",
                f, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
                snn.best_val_r2, len(snn.history),
            )
            rows.append({
                "model": "trained_snn",
                "event_budget": float(f), "seed": int(seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"], "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "r2_vx_ci_lo": r2_boot["r2_vx_ci_lo"], "r2_vx_ci_hi": r2_boot["r2_vx_ci_hi"],
                "r2_vy_ci_lo": r2_boot["r2_vy_ci_lo"], "r2_vy_ci_hi": r2_boot["r2_vy_ci_hi"],
                "n_boot": int(args.n_boot),
                "best_val_r2": float(snn.best_val_r2),
                "epochs_trained": int(len(snn.history)),
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": "BPTT + fast-sigmoid surrogate",
            })
            append_result(args.results_csv, rows[-1])

            if seed == QUALITATIVE_SEED and abs(f - QUALITATIVE_BUDGET) < 1e-9:
                pred_path = args.results_csv.parent / f"predictions_f{f:.2f}_seed{seed}.npz"
                np.savez(
                    pred_path,
                    y_true=y[test_idx], y_pred=y_pred.astype(np.float32),
                    test_idx=test_idx, bin_size_ms=np.array(int(bin_size_ms)),
                    event_budget=np.array(float(f)), model=np.array("trained_snn"),
                )
                logger.info("saved predictions to %s", pred_path)

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms,
        "num_neurons": num_neurons,
        "hidden_dim": int(args.hidden_dim),
        "tau_ms": float(args.tau_ms),
        "threshold": float(args.threshold),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "epochs_max": int(args.epochs),
        "patience": int(args.patience),
        "surrogate_slope": float(args.surrogate_slope),
        "event_budgets": list(args.event_budgets),
        "seeds": list(args.seeds),
        "n_boot": int(args.n_boot),
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
    }
    save_json_results(args.results_json, model="trained_snn", config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
