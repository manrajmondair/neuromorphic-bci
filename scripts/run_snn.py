"""Train and evaluate the sparse spike-latency SNN and the order-shuffle control.

For each (seed, event_budget):
  1. Filter the processed dataset to the earliest fraction f of spike
     events per bin.
  2. Tune the LIF threshold on the val split (per-budget) so the hidden
     layer fires at a useful rate for that event density.
  3. Fit the SparseLatencySNN with multi-restart random projections and
     ridge readout (best-of-n_restarts kept).
  4. Score on the test split, with bootstrap 95% CIs.
  5. Repeat for the order-shuffle control (same retained events, within-bin
     order permuted) using the threshold tuned on the real-order SNN.

Writes the canonical per-model JSON tracking files + flat CSVs + per-bin
prediction npzs at f=0.25, seed=0 for the qualitative panel.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.controls.order_shuffle import shuffle_within_bin_order
from src.data.preprocess import load_processed
from src.evaluation.experiment_runner import append_result, save_json_results
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import apply_event_budget
from src.models.snn_decoder import SparseLatencySNN, tune_threshold_on_val
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_snn")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)
SEEDS_DEFAULT = (0, 1, 2)
THRESHOLDS_DEFAULT = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
QUALITATIVE_BUDGET = 0.25
QUALITATIVE_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SNN baseline and order-shuffle control across event budgets.")
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--snn-csv", type=Path, default=Path("results/snn/results.csv"))
    p.add_argument("--snn-json", type=Path, default=Path("results/snn/snn_results.json"))
    p.add_argument("--shuffle-csv", type=Path, default=Path("results/controls/results.csv"))
    p.add_argument("--shuffle-json", type=Path, default=Path("results/controls/shuffle_results.json"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=list(EVENT_BUDGETS_DEFAULT))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument(
        "--thresholds", type=float, nargs="+", default=list(THRESHOLDS_DEFAULT),
        help="threshold grid swept on val split per budget",
    )
    p.add_argument("--fixed-threshold", type=float, default=None,
                   help="skip tuning, use this threshold everywhere")
    p.add_argument("--readout-alpha", type=float, default=1.0)
    p.add_argument("--n-restarts", type=int, default=3,
                   help="independent W init restarts per (seed, budget); best val R^2 wins")
    p.add_argument("--no-standardize", action="store_true",
                   help="skip per-hidden-unit z-scoring before the readout")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def _save_predictions(path, y_true, y_pred, test_idx, bin_size_ms, f, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        y_true=y_true.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        test_idx=test_idx,
        bin_size_ms=np.array(int(bin_size_ms)),
        event_budget=np.array(float(f)),
        model=np.array(model),
    )
    logger.info("saved predictions to %s", path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    data = load_processed(args.processed_path)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    y = np.asarray(data["velocity"], dtype=np.float32)
    num_bins, num_neurons = data["spike_counts"].shape
    bin_size_ms = int(data["bin_size_ms"])
    logger.info(
        "loaded processed data: %d bins, %d neurons (train=%d val=%d test=%d)",
        num_bins, num_neurons, train_idx.size, val_idx.size, test_idx.size,
    )

    for csv_path in (args.snn_csv, args.shuffle_csv):
        if csv_path.exists():
            csv_path.unlink()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_events_total = int(sum(t.size for t in data["event_times"]))
    snn_rows: list[dict] = []
    shuffle_rows: list[dict] = []

    for seed in args.seeds:
        set_global_seed(seed)
        for f in args.event_budgets:
            logger.info("=" * 72)
            logger.info("snn: seed=%d event_budget=%.2f", seed, f)
            logger.info("=" * 72)
            et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
            n_events_used = int(sum(t.size for t in et))

            # Threshold tuning on val for this budget (skip if fixed-threshold given).
            if args.fixed_threshold is None:
                best_thr, sweep = tune_threshold_on_val(
                    num_neurons=num_neurons,
                    bin_size_ms=bin_size_ms,
                    event_times=et, event_neurons=en,
                    velocity=y, train_idx=train_idx, val_idx=val_idx,
                    thresholds=tuple(args.thresholds),
                    hidden_dim=args.hidden_dim,
                    tau_ms=args.tau_ms,
                    readout_alpha=args.readout_alpha,
                    n_restarts=1, standardize=not args.no_standardize, seed=seed,
                )
                logger.info("threshold sweep at f=%.2f: %s -> best=%g", f, sweep, best_thr)
            else:
                best_thr = float(args.fixed_threshold)
                sweep = None

            # --- real-order SNN ---
            snn = SparseLatencySNN(
                num_neurons=num_neurons,
                hidden_dim=args.hidden_dim,
                tau_ms=args.tau_ms,
                threshold=best_thr,
                readout_alpha=args.readout_alpha,
                bin_size_ms=bin_size_ms,
                n_restarts=args.n_restarts,
                standardize=not args.no_standardize,
                seed=seed,
            ).fit(et, en, y, train_idx, val_idx)
            y_pred = snn.predict(et, en, test_idx)
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
            logger.info(
                "snn result: f=%.2f seed=%d  r2_joint=%+.4f [%.4f, %.4f]  thr=%g restarts_val=%s",
                f, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
                best_thr, [round(s, 4) for s in snn.restart_val_r2s],
            )
            snn_rows.append({
                "model": "snn",
                "event_budget": float(f), "seed": int(seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"], "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "r2_vx_ci_lo": r2_boot["r2_vx_ci_lo"], "r2_vx_ci_hi": r2_boot["r2_vx_ci_hi"],
                "r2_vy_ci_lo": r2_boot["r2_vy_ci_lo"], "r2_vy_ci_hi": r2_boot["r2_vy_ci_hi"],
                "n_boot": int(args.n_boot),
                "tuned_threshold": float(best_thr),
                "threshold_sweep": sweep,
                "restart_val_r2s": snn.restart_val_r2s,
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": "",
            })
            append_result(args.snn_csv, snn_rows[-1])
            if seed == QUALITATIVE_SEED and abs(f - QUALITATIVE_BUDGET) < 1e-9:
                _save_predictions(
                    args.snn_csv.parent / f"predictions_f{f:.2f}_seed{seed}.npz",
                    y[test_idx], y_pred, test_idx, bin_size_ms, f, "snn",
                )

            # --- order-shuffle control: same threshold, same hyperparameters ---
            et_s, en_s = shuffle_within_bin_order(et, en, seed=seed)
            snn_s = SparseLatencySNN(
                num_neurons=num_neurons,
                hidden_dim=args.hidden_dim,
                tau_ms=args.tau_ms,
                threshold=best_thr,
                readout_alpha=args.readout_alpha,
                bin_size_ms=bin_size_ms,
                n_restarts=args.n_restarts,
                standardize=not args.no_standardize,
                seed=seed,
            ).fit(et_s, en_s, y, train_idx, val_idx)
            y_pred_s = snn_s.predict(et_s, en_s, test_idx)
            r2_s = velocity_r2(y[test_idx], y_pred_s)
            r2_s_boot = velocity_r2_bootstrap(y[test_idx], y_pred_s, n_boot=args.n_boot, seed=seed)
            logger.info(
                "shuffle result: f=%.2f seed=%d  r2_joint=%+.4f [%.4f, %.4f]",
                f, seed, r2_s["r2_joint"],
                r2_s_boot["r2_joint_ci_lo"], r2_s_boot["r2_joint_ci_hi"],
            )
            shuffle_rows.append({
                "model": "snn_shuffle",
                "event_budget": float(f), "seed": int(seed),
                "r2_vx": r2_s["r2_vx"], "r2_vy": r2_s["r2_vy"], "r2_joint": r2_s["r2_joint"],
                "r2_joint_ci_lo": r2_s_boot["r2_joint_ci_lo"], "r2_joint_ci_hi": r2_s_boot["r2_joint_ci_hi"],
                "r2_vx_ci_lo": r2_s_boot["r2_vx_ci_lo"], "r2_vx_ci_hi": r2_s_boot["r2_vx_ci_hi"],
                "r2_vy_ci_lo": r2_s_boot["r2_vy_ci_lo"], "r2_vy_ci_hi": r2_s_boot["r2_vy_ci_hi"],
                "n_boot": int(args.n_boot),
                "tuned_threshold": float(best_thr),
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": "within-bin order permuted",
            })
            append_result(args.shuffle_csv, shuffle_rows[-1])
            if seed == QUALITATIVE_SEED and abs(f - QUALITATIVE_BUDGET) < 1e-9:
                _save_predictions(
                    args.shuffle_csv.parent / f"predictions_f{f:.2f}_seed{seed}.npz",
                    y[test_idx], y_pred_s, test_idx, bin_size_ms, f, "snn_shuffle",
                )

    common_config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms,
        "num_neurons": int(num_neurons),
        "hidden_dim": int(args.hidden_dim),
        "tau_ms": float(args.tau_ms),
        "thresholds_swept": list(args.thresholds) if args.fixed_threshold is None else None,
        "fixed_threshold": (None if args.fixed_threshold is None else float(args.fixed_threshold)),
        "readout_alpha": float(args.readout_alpha),
        "n_restarts": int(args.n_restarts),
        "standardize": not args.no_standardize,
        "event_budgets": list(args.event_budgets),
        "seeds": list(args.seeds),
        "n_boot": int(args.n_boot),
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
    }
    save_json_results(args.snn_json, model="snn", config=common_config, rows=snn_rows)
    save_json_results(args.shuffle_json, model="snn_shuffle", config=common_config, rows=shuffle_rows)
    logger.info("wrote %s and %s", args.snn_csv, args.snn_json)
    logger.info("wrote %s and %s", args.shuffle_csv, args.shuffle_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
