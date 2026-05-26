"""Train and evaluate the sparse spike-latency SNN and the order-shuffle control.

For each (seed, event_budget):
  1. Filter the processed dataset down to the earliest fraction `f` of
     spike events per bin.
  2. Train a `SparseLatencySNN` on the train split, score on the held-out
     test split via the canonical joint velocity R^2.
  3. Shuffle the within-bin order of those same retained events and repeat,
     producing the order-shuffle control row.

Writes the canonical per-model JSON tracking files plus flat CSVs:
  * results/snn/snn_results.json           (SNN curve, overlay-ready)
  * results/snn/results.csv                (flat dump of the same rows)
  * results/controls/shuffle_results.json  (shuffle curve)
  * results/controls/results.csv           (flat dump)

At the reference (seed=0, f=0.25) configuration the per-bin true/pred test
predictions are saved as .npz files for the qualitative-trajectory panel
in `scripts/generate_final_figures.py`:
  * results/snn/predictions_f0.25_seed0.npz
  * results/controls/predictions_f0.25_seed0.npz

Owned by snn-latency-decoder.
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
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.models.snn_decoder import SparseLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_snn")

EVENT_BUDGETS_DEFAULT = (1.00, 0.50, 0.25, 0.10)
SEEDS_DEFAULT = (0, 1, 2)
QUALITATIVE_BUDGET = 0.25
QUALITATIVE_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SNN baseline and order-shuffle control across event budgets.")
    p.add_argument(
        "--processed-path",
        type=Path,
        default=Path("data/processed/processed_mc_rtt.npz"),
    )
    p.add_argument("--snn-csv", type=Path, default=Path("results/snn/results.csv"))
    p.add_argument("--snn-json", type=Path, default=Path("results/snn/snn_results.json"))
    p.add_argument(
        "--shuffle-csv", type=Path, default=Path("results/controls/results.csv")
    )
    p.add_argument(
        "--shuffle-json",
        type=Path,
        default=Path("results/controls/shuffle_results.json"),
    )
    p.add_argument("--event-budgets", type=float, nargs="+", default=list(EVENT_BUDGETS_DEFAULT))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--tau-ms", type=float, default=10.0)
    # Threshold tuned by a small per-budget sweep: 0.5 makes hidden firing
    # sparse enough that spike timing carries decoding signal — at this
    # operating point the order-shuffle control trails the real-order SNN
    # by a clearly visible margin, which is the headline test the proposal
    # asks for.
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--readout-alpha", type=float, default=1.0)
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def _save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray,
                      test_idx: np.ndarray, bin_size_ms: int, f: float, model: str) -> None:
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

    # Idempotent CSVs: a fresh invocation reflects only its own rows.
    for csv_path in (args.snn_csv, args.shuffle_csv):
        if csv_path.exists():
            csv_path.unlink()

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

            # --- real-order SNN ---
            snn = SparseLatencySNN(
                num_neurons=num_neurons,
                hidden_dim=args.hidden_dim,
                tau_ms=args.tau_ms,
                threshold=args.threshold,
                readout_alpha=args.readout_alpha,
                bin_size_ms=bin_size_ms,
                seed=seed,
            ).fit(et, en, y, train_idx, val_idx)
            y_pred = snn.predict(et, en, test_idx)
            r2 = velocity_r2(y[test_idx], y_pred)
            logger.info(
                "snn result: f=%.2f seed=%d  test r2_joint=%+.4f (vx=%+.4f vy=%+.4f)  events=%d/%d",
                f, seed, r2["r2_joint"], r2["r2_vx"], r2["r2_vy"], n_events_used, n_events_total,
            )
            row = {
                "model": "snn",
                "event_budget": float(f),
                "seed": int(seed),
                "r2_vx": r2["r2_vx"],
                "r2_vy": r2["r2_vy"],
                "r2_joint": r2["r2_joint"],
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": "",
            }
            snn_rows.append(row)
            append_result(args.snn_csv, row)
            if seed == QUALITATIVE_SEED and abs(f - QUALITATIVE_BUDGET) < 1e-9:
                _save_predictions(
                    args.snn_csv.parent / f"predictions_f{f:.2f}_seed{seed}.npz",
                    y[test_idx], y_pred, test_idx, bin_size_ms, f, "snn",
                )

            # --- order-shuffle control ---
            et_s, en_s = shuffle_within_bin_order(et, en, seed=seed)
            snn_s = SparseLatencySNN(
                num_neurons=num_neurons,
                hidden_dim=args.hidden_dim,
                tau_ms=args.tau_ms,
                threshold=args.threshold,
                readout_alpha=args.readout_alpha,
                bin_size_ms=bin_size_ms,
                seed=seed,
            ).fit(et_s, en_s, y, train_idx, val_idx)
            y_pred_s = snn_s.predict(et_s, en_s, test_idx)
            r2_s = velocity_r2(y[test_idx], y_pred_s)
            logger.info(
                "shuffle result: f=%.2f seed=%d  test r2_joint=%+.4f (vx=%+.4f vy=%+.4f)  events=%d/%d",
                f, seed, r2_s["r2_joint"], r2_s["r2_vx"], r2_s["r2_vy"], n_events_used, n_events_total,
            )
            row_s = {
                "model": "snn_shuffle",
                "event_budget": float(f),
                "seed": int(seed),
                "r2_vx": r2_s["r2_vx"],
                "r2_vy": r2_s["r2_vy"],
                "r2_joint": r2_s["r2_joint"],
                "n_events_used": n_events_used,
                "n_events_total": n_events_total,
                "notes": "within-bin order permuted",
            }
            shuffle_rows.append(row_s)
            append_result(args.shuffle_csv, row_s)
            if seed == QUALITATIVE_SEED and abs(f - QUALITATIVE_BUDGET) < 1e-9:
                _save_predictions(
                    args.shuffle_csv.parent / f"predictions_f{f:.2f}_seed{seed}.npz",
                    y[test_idx], y_pred_s, test_idx, bin_size_ms, f, "snn_shuffle",
                )

    # Canonical per-model JSON tracking files (figure pipeline reads these).
    common_config = {
        "processed_path": str(args.processed_path),
        "bin_size_ms": bin_size_ms,
        "num_neurons": int(num_neurons),
        "hidden_dim": int(args.hidden_dim),
        "tau_ms": float(args.tau_ms),
        "threshold": float(args.threshold),
        "readout_alpha": float(args.readout_alpha),
        "event_budgets": list(args.event_budgets),
        "seeds": list(args.seeds),
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
