"""SNN readout ablation — compare ridge / SGD-linear / MLP heads on the same reservoir.

For each event budget at seed 0, the SNN encoder is fit once with the
per-budget threshold tuning + multi-restart pipeline. The resulting
hidden spike-count matrix Z is then handed to each of three readouts:

  * `ridge`        — closed-form Tikhonov ridge regression
  * `sgd_linear`   — same hypothesis class as ridge, trained by Adam +
                     early stopping on val
  * `mlp`          — small 2-layer ReLU MLP, Adam + early stopping

Each readout is scored on the test split with bootstrap 95% CIs.

Writes:
  * results/snn/readout_ablation.json  (canonical schema with one row per readout)
  * results/snn/readout_ablation.csv
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
from src.models.readout import LinearReadout, MLPReadout, SGDLinearReadout
from src.models.snn_decoder import SparseLatencySNN, tune_threshold_on_val
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_readout_ablation")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/snn/readout_ablation.csv"))
    p.add_argument("--results-json", type=Path, default=Path("results/snn/readout_ablation.json"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=list(EVENT_BUDGETS_DEFAULT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _encode_for_budget(data, f, seed, hidden_dim, tau_ms, n_restarts):
    et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
    best_thr, _ = tune_threshold_on_val(
        num_neurons=int(data["num_neurons"]),
        bin_size_ms=int(data["bin_size_ms"]),
        event_times=et, event_neurons=en,
        velocity=np.asarray(data["velocity"], dtype=np.float32),
        train_idx=data["train_idx"], val_idx=data["val_idx"],
        hidden_dim=hidden_dim, tau_ms=tau_ms, n_restarts=1, seed=seed,
    )
    snn = SparseLatencySNN(
        num_neurons=int(data["num_neurons"]),
        hidden_dim=hidden_dim, tau_ms=tau_ms, threshold=best_thr,
        bin_size_ms=int(data["bin_size_ms"]),
        n_restarts=n_restarts, standardize=True, seed=seed,
    ).fit(et, en, np.asarray(data["velocity"], dtype=np.float32),
          data["train_idx"], data["val_idx"])
    # Encode all bins with the fitted W + standardization
    Z = snn._encode_with_W(snn.W, et, en)
    Z = (Z - snn.mu) / snn.sigma
    return Z, best_thr


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    if args.results_csv.exists():
        args.results_csv.unlink()
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    set_global_seed(args.seed)

    for f in args.event_budgets:
        logger.info("=" * 72)
        logger.info("encoding SNN at f=%.2f", f)
        logger.info("=" * 72)
        Z, thr = _encode_for_budget(
            data, f, args.seed, args.hidden_dim, args.tau_ms, args.n_restarts,
        )

        readout_classes = {
            "ridge": lambda: LinearReadout(alpha=1.0),
            "sgd_linear": lambda: SGDLinearReadout(seed=args.seed),
            "mlp": lambda: MLPReadout(hidden_dim=64, seed=args.seed),
        }
        for name, ctor in readout_classes.items():
            readout = ctor()
            readout.fit(Z[train_idx], y[train_idx], Z[val_idx], y[val_idx])
            y_pred = readout.predict(Z[test_idx])
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=args.seed)
            logger.info(
                "%-11s f=%.2f thr=%g  r2_joint=%+.4f [%.4f, %.4f]",
                name, f, thr, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
            )
            row = {
                "model": f"snn_readout_{name}",
                "readout": name,
                "event_budget": float(f),
                "seed": int(args.seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
                "tuned_threshold": float(thr),
                "notes": "snn encoder fixed; readout swapped",
            }
            rows.append(row)
            append_result(args.results_csv, row)

    config = {
        "processed_path": str(args.processed_path),
        "event_budgets": list(args.event_budgets),
        "seed": int(args.seed),
        "hidden_dim": int(args.hidden_dim),
        "tau_ms": float(args.tau_ms),
        "n_restarts": int(args.n_restarts),
        "n_boot": int(args.n_boot),
    }
    save_json_results(args.results_json, model="snn_readout_ablation", config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
