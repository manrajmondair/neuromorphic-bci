"""Temporal generalization — chronic-BCI drift stand-in for cross-subject transfer.

MC_RTT has one subject (Indy) so a true cross-subject experiment is
unavailable. The natural within-recording proxy is: does a decoder
trained on *older* training data perform worse on the held-out test
split than one trained on *newer* training data? If yes, the neural
population is drifting and chronic deployment would degrade.

For each model in {ridge, reservoir SNN} we fit three variants:
  * `train_early`  — only the first 50% of train_idx (temporally distant from test)
  * `train_late`   — only the last 50% of train_idx (closer to test)
  * `train_full`   — all of train_idx (the canonical run, included for reference)

All three are evaluated on the SAME standard test_idx. Each row carries
bootstrap CIs. The interesting quantity is (full - early) vs (full - late):
if the early-trained decoder loses more accuracy than the late-trained
one, that's measurable drift.

Writes results/temporal/temporal_generalization.json.
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
from src.features.spike_counts import stack_lag_features
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.models.snn_decoder import SparseLatencySNN, tune_threshold_on_val
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_temporal_generalization")


def _slice_train(train_idx: np.ndarray, regime: str) -> np.ndarray:
    n = train_idx.size
    if regime == "full":
        return train_idx
    if regime == "early":
        return train_idx[: n // 2]
    if regime == "late":
        return train_idx[n // 2 :]
    raise ValueError(f"unknown regime {regime!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--results-csv", type=Path, default=Path("results/temporal/results.csv"))
    p.add_argument("--results-json", type=Path,
                   default=Path("results/temporal/temporal_generalization.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-restarts", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=300)
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
    n_events_total = int(sum(t.size for t in data["event_times"]))

    rows = []
    for seed in args.seeds:
        for regime in ("early", "late", "full"):
            tr = _slice_train(train_idx, regime)
            set_global_seed(seed)
            logger.info("=" * 72)
            logger.info("temporal: seed=%d regime=%s  train=%d val=%d test=%d",
                        seed, regime, tr.size, val_idx.size, test_idx.size)
            logger.info("=" * 72)

            # Single-bin ridge
            X = stack_lag_features(
                data["spike_counts"].astype(np.float32), num_lags=0,
            )
            decoder = RidgeDecoder(alphas=DEFAULT_ALPHAS).fit(
                X[tr], y[tr], X[val_idx], y[val_idx],
            )
            y_pred_r = decoder.predict(X[test_idx])
            r2_r = velocity_r2(y[test_idx], y_pred_r)
            r2_r_boot = velocity_r2_bootstrap(y[test_idx], y_pred_r,
                                              n_boot=args.n_boot, seed=seed)
            row = {
                "model": "ridge", "regime": regime,
                "train_size": int(tr.size),
                "seed": int(seed),
                "r2_vx": r2_r["r2_vx"], "r2_vy": r2_r["r2_vy"], "r2_joint": r2_r["r2_joint"],
                "r2_joint_ci_lo": r2_r_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_r_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
                "best_alpha": decoder.best_alpha,
                "n_events_used": n_events_total,
                "n_events_total": n_events_total,
                "notes": f"ridge trained on {regime} half",
            }
            rows.append(row)
            append_result(args.results_csv, row)
            logger.info("ridge  %s seed=%d  r2_joint=%+.4f [%.4f, %.4f]",
                        regime, seed, r2_r["r2_joint"],
                        r2_r_boot["r2_joint_ci_lo"], r2_r_boot["r2_joint_ci_hi"])

            # Reservoir SNN (real-order events) on full event budget
            et = data["event_times"]
            en = data["event_neurons"]
            best_thr, _ = tune_threshold_on_val(
                num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                event_times=et, event_neurons=en,
                velocity=y, train_idx=tr, val_idx=val_idx,
                hidden_dim=args.hidden_dim, n_restarts=1, seed=seed,
            )
            snn = SparseLatencySNN(
                num_neurons=num_neurons, hidden_dim=args.hidden_dim,
                threshold=best_thr, bin_size_ms=bin_size_ms,
                n_restarts=args.n_restarts, standardize=True, seed=seed,
            ).fit(et, en, y, tr, val_idx)
            y_pred_s = snn.predict(et, en, test_idx)
            r2_s = velocity_r2(y[test_idx], y_pred_s)
            r2_s_boot = velocity_r2_bootstrap(y[test_idx], y_pred_s,
                                              n_boot=args.n_boot, seed=seed)
            row = {
                "model": "snn", "regime": regime,
                "train_size": int(tr.size),
                "seed": int(seed),
                "r2_vx": r2_s["r2_vx"], "r2_vy": r2_s["r2_vy"], "r2_joint": r2_s["r2_joint"],
                "r2_joint_ci_lo": r2_s_boot["r2_joint_ci_lo"],
                "r2_joint_ci_hi": r2_s_boot["r2_joint_ci_hi"],
                "n_boot": int(args.n_boot),
                "tuned_threshold": float(best_thr),
                "notes": f"reservoir SNN trained on {regime} half",
            }
            rows.append(row)
            append_result(args.results_csv, row)
            logger.info("snn    %s seed=%d  r2_joint=%+.4f [%.4f, %.4f]",
                        regime, seed, r2_s["r2_joint"],
                        r2_s_boot["r2_joint_ci_lo"], r2_s_boot["r2_joint_ci_hi"])

    config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms, "num_neurons": num_neurons,
        "seeds": list(args.seeds),
        "hidden_dim": int(args.hidden_dim),
        "n_restarts": int(args.n_restarts),
        "n_boot": int(args.n_boot),
        "regimes": ["early", "late", "full"],
        "notes": (
            "MC_RTT has one subject; this is a within-recording temporal-drift stand-in. "
            "Early = first half of train_idx; late = second half; full = all of train_idx. "
            "All three evaluated on the same canonical test_idx."
        ),
    }
    save_json_results(args.results_json, model="temporal_generalization",
                      config=config, rows=rows)
    logger.info("wrote %s and %s", args.results_csv, args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
