"""Causal decoding window sweep — accuracy as a function of in-bin lookahead.

For each window_ms in {10, 20, 30, 40, 50} ms we drop every within-bin
event whose time is >= window_ms, then refit the reservoir SNN and a
single-bin ridge for direct comparison. Smaller windows simulate the
practical latency of a deployed BCI that has to predict during the
bin.

Writes results/causal/causal_window_results.json.
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
from src.features.causal_window import truncate_to_window
from src.features.spike_counts import counts_from_events
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.models.snn_decoder import SparseLatencySNN, tune_threshold_on_val
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_causal_window")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/causal/results.csv"))
    p.add_argument("--results-json", type=Path, default=Path("results/causal/causal_window_results.json"))
    p.add_argument("--windows-ms", type=float, nargs="+", default=[10.0, 20.0, 30.0, 40.0, 50.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-restarts", type=int, default=2)
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
    for window_ms in args.windows_ms:
        logger.info("=" * 72)
        logger.info("causal window: %.1f ms", window_ms)
        logger.info("=" * 72)
        et, en = truncate_to_window(
            data["event_times"], data["event_neurons"], window_ms=window_ms,
        )
        spike_counts = counts_from_events(en, num_neurons).astype(np.float32)

        for seed in args.seeds:
            set_global_seed(seed)

            # Single-bin ridge on truncated counts
            decoder = RidgeDecoder(alphas=DEFAULT_ALPHAS).fit(
                spike_counts[train_idx], y[train_idx],
                spike_counts[val_idx], y[val_idx],
            )
            y_ridge = decoder.predict(spike_counts[test_idx])
            r2_r = velocity_r2(y[test_idx], y_ridge)
            r2_r_boot = velocity_r2_bootstrap(y[test_idx], y_ridge, n_boot=args.n_boot, seed=seed)
            rows.append({
                "model": "ridge_causal",
                "window_ms": float(window_ms),
                "event_budget": 1.00, "seed": int(seed),
                "r2_vx": r2_r["r2_vx"], "r2_vy": r2_r["r2_vy"], "r2_joint": r2_r["r2_joint"],
                "r2_joint_ci_lo": r2_r_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_r_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
                "best_alpha": decoder.best_alpha,
                "notes": "single-bin ridge on causally-truncated counts",
            })
            append_result(args.results_csv, rows[-1])
            logger.info(
                "ridge w=%.0f ms seed=%d  r2_joint=%+.4f [%.4f, %.4f]",
                window_ms, seed, r2_r["r2_joint"],
                r2_r_boot["r2_joint_ci_lo"], r2_r_boot["r2_joint_ci_hi"],
            )

            # SNN on truncated events
            best_thr, _ = tune_threshold_on_val(
                num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                event_times=et, event_neurons=en,
                velocity=y, train_idx=train_idx, val_idx=val_idx,
                hidden_dim=args.hidden_dim, n_restarts=1, seed=seed,
            )
            snn = SparseLatencySNN(
                num_neurons=num_neurons, hidden_dim=args.hidden_dim,
                threshold=best_thr, bin_size_ms=bin_size_ms,
                n_restarts=args.n_restarts, standardize=True, seed=seed,
            ).fit(et, en, y, train_idx, val_idx)
            y_snn = snn.predict(et, en, test_idx)
            r2 = velocity_r2(y[test_idx], y_snn)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_snn, n_boot=args.n_boot, seed=seed)
            rows.append({
                "model": "snn_causal",
                "window_ms": float(window_ms),
                "event_budget": 1.00, "seed": int(seed),
                "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
                "tuned_threshold": float(best_thr),
                "notes": "reservoir SNN on causally-truncated events",
            })
            append_result(args.results_csv, rows[-1])
            logger.info(
                "snn   w=%.0f ms seed=%d  r2_joint=%+.4f [%.4f, %.4f]",
                window_ms, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
            )

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms, "num_neurons": num_neurons,
        "windows_ms": list(args.windows_ms),
        "seeds": list(args.seeds),
        "hidden_dim": int(args.hidden_dim),
        "n_restarts": int(args.n_restarts),
        "n_boot": int(args.n_boot),
    }
    save_json_results(args.results_json, model="causal_window", config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
