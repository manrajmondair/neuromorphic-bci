"""Train and evaluate the deeper / recurrent trained SNN.

For each (seed, event_budget) we fit a `DeeperTrainedSNN` (two or
three stacked LIF layers, optional within-layer recurrence) and score
on the test split with bootstrap CIs. Run with `--recurrent` to add a
recurrent weight to each layer.

The default config is 2 LIF layers (128 → 64) without recurrence —
matches the single-layer trained SNN's parameter count but lets the
encoder build a deeper temporal feature.

Writes results/deeper_snn/.
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
from src.models.deeper_snn import DeeperTrainedSNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_deeper_snn")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/deeper_snn/results.csv"))
    p.add_argument("--results-json", type=Path, default=Path("results/deeper_snn/deeper_snn_results.json"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.50, 0.25, 0.10])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--recurrent", action="store_true")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    if args.results_csv.exists():
        args.results_csv.unlink()
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])

    rows = []
    for seed in args.seeds:
        set_global_seed(seed)
        for f in args.event_budgets:
            et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
            logger.info("=" * 72)
            logger.info("deeper_snn: seed=%d budget=%.2f layers=%s recurrent=%s",
                        seed, f, args.hidden_dims, args.recurrent)
            logger.info("=" * 72)
            snn = DeeperTrainedSNN(
                num_neurons=num_neurons, hidden_dims=tuple(args.hidden_dims),
                tau_ms=args.tau_ms, threshold=args.threshold,
                bin_size_ms=bin_size_ms, k_history=args.k_history,
                recurrent=args.recurrent,
                lr=args.lr, weight_decay=args.weight_decay,
                epochs=args.epochs, patience=args.patience, seed=seed,
            ).fit(et, en, y, train_idx, val_idx)
            split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
            y_pred = snn.predict(et, en, test_idx, split_starts=split_starts)
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
            logger.info(
                "deeper_snn result: f=%.2f seed=%d  r2_joint=%+.4f [%.4f, %.4f] val=%+.4f epochs=%d",
                f, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
                snn.best_val_r2, len(snn.history),
            )
            model_name = ("deeper_snn_recurrent" if args.recurrent else "deeper_snn")
            row = {
                "model": model_name,
                "event_budget": float(f), "seed": int(seed),
                "hidden_dims": list(args.hidden_dims),
                "recurrent": bool(args.recurrent),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
                "best_val_r2": float(snn.best_val_r2),
            }
            rows.append(row)
            append_result(args.results_csv, row)

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms, "num_neurons": num_neurons,
        "hidden_dims": list(args.hidden_dims),
        "recurrent": bool(args.recurrent),
        "k_history": int(args.k_history),
        "threshold": float(args.threshold),
        "tau_ms": float(args.tau_ms),
        "lr": float(args.lr), "weight_decay": float(args.weight_decay),
        "epochs_max": int(args.epochs), "patience": int(args.patience),
        "event_budgets": list(args.event_budgets), "seeds": list(args.seeds),
        "n_boot": int(args.n_boot),
    }
    model_top = "deeper_snn_recurrent" if args.recurrent else "deeper_snn"
    save_json_results(args.results_json, model=model_top, config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
