"""Train and evaluate the pure-latency ridge decoder across event budgets.

Feature: time-to-first-spike per neuron (with silent neurons at the
bin_size_ms sentinel). Readout: ridge regression with an alpha sweep on
the val split. Bootstrap CIs on test R^2.

Writes:
  * results/latency/latency_results.json     canonical schema
  * results/latency/results.csv              flat dump
  * results/latency/predictions_f{0.25}_seed0.npz   for trajectory figure

Owned by snn-latency-decoder territory but mirrors the ridge / snn
script layout so the figure pipeline picks it up without code changes.
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
from src.models.latency_decoder import LatencyDecoder
from src.models.ridge_decoder import DEFAULT_ALPHAS
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_latency")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)
SEEDS_DEFAULT = (0,)
QUALITATIVE_BUDGET = 0.25
QUALITATIVE_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pure-latency ridge decoder across event budgets.")
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/latency/results.csv"))
    p.add_argument("--results-json", type=Path, default=Path("results/latency/latency_results.json"))
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument("--event-budgets", type=float, nargs="+", default=list(EVENT_BUDGETS_DEFAULT))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-standardize", action="store_true", help="skip per-neuron mean/std normalization")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)

    data = load_processed(args.processed_path)
    if args.results_csv.exists():
        args.results_csv.unlink()
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    y = np.asarray(data["velocity"], dtype=np.float32)
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])
    n_events_total = int(sum(t.size for t in data["event_times"]))

    rows: list[dict] = []
    for seed in args.seeds:
        set_global_seed(seed)
        for f in args.event_budgets:
            logger.info("=" * 72)
            logger.info("latency: seed=%d event_budget=%.2f", seed, f)
            logger.info("=" * 72)
            et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)

            decoder = LatencyDecoder(
                num_neurons=num_neurons,
                bin_size_ms=bin_size_ms,
                standardize=not args.no_standardize,
            ).fit(et, en, y, train_idx, val_idx, alphas=tuple(args.alphas))
            y_pred = decoder.predict(et, en, test_idx)
            r2 = velocity_r2(y[test_idx], y_pred)
            r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
            n_events_used = int(sum(t.size for t in et))

            logger.info(
                "latency result: f=%.2f seed=%d  r2_joint=%+.4f [%.4f, %.4f]  alpha=%g  events=%d/%d",
                f, seed, r2["r2_joint"],
                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"],
                decoder.best_alpha, n_events_used, n_events_total,
            )

            row = {
                "model": "latency",
                "event_budget": float(f),
                "seed": int(seed),
                "r2_vx": r2["r2_vx"],
                "r2_vy": r2["r2_vy"],
                "r2_joint": r2["r2_joint"],
                "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                "r2_vx_ci_lo": r2_boot["r2_vx_ci_lo"],
                "r2_vx_ci_hi": r2_boot["r2_vx_ci_hi"],
                "r2_vy_ci_lo": r2_boot["r2_vy_ci_lo"],
                "r2_vy_ci_hi": r2_boot["r2_vy_ci_hi"],
                "n_boot": int(args.n_boot),
                "best_alpha": decoder.best_alpha,
                "alpha_sweep": decoder.alpha_sweep,
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": "first-spike latency features only",
            }
            rows.append(row)
            append_result(args.results_csv, row)

            if seed == QUALITATIVE_SEED and abs(f - QUALITATIVE_BUDGET) < 1e-9:
                pred_path = args.results_csv.parent / f"predictions_f{f:.2f}_seed{seed}.npz"
                np.savez(
                    pred_path,
                    y_true=y[test_idx],
                    y_pred=y_pred.astype(np.float32),
                    test_idx=test_idx,
                    bin_size_ms=np.array(int(bin_size_ms)),
                    event_budget=np.array(float(f)),
                    model=np.array("latency"),
                )
                logger.info("saved predictions to %s", pred_path)

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms,
        "num_neurons": num_neurons,
        "event_budgets": list(args.event_budgets),
        "seeds": list(args.seeds),
        "alphas_swept": list(args.alphas),
        "standardize": not args.no_standardize,
        "n_boot": int(args.n_boot),
        "split_sizes": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
    }
    save_json_results(args.results_json, model="latency", config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
