"""Energy-accuracy Pareto frontier for the trained SNN.

Sweeps a (hidden_dim, k_history) grid; for each config, trains the
TrainedLatencySNN, measures test-set R^2, AND counts the actual
input/hidden synaptic operations on the test split. The combined
(R^2, synops_per_prediction) cloud lets us draw a Pareto frontier
relating per-prediction energy (on Loihi 2 / NorthPole) to decoding
accuracy — the headline neuromorphic figure.

Writes `results/cluster/pareto/pareto_energy_accuracy.json`.

Re-uses the synop counters from `scripts/run_loihi_estimate.py` so the
energy numbers are directly comparable to the single-point estimate at
the canonical h=128, k_history=4 setting.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.preprocess import load_processed
from src.evaluation.efficiency_tracker import MAC_ENERGY_PJ
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.models.trained_snn import (
    TrainedLatencySNN,
    _sparse_events_to_subbin_counts,
    _stack_history,
)
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_pareto_energy_accuracy")


def _count_synops(
    snn: TrainedLatencySNN,
    et: list[np.ndarray],
    en: list[np.ndarray],
    test_idx: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_neurons: int,
    bin_size_ms: int,
    num_sub_bins: int,
    k_history: int,
) -> dict:
    """Input + hidden synop counts on the test split. Mirrors run_loihi_estimate.py."""
    sub_counts = _sparse_events_to_subbin_counts(
        et, en, num_neurons=num_neurons, bin_size_ms=bin_size_ms, num_sub_bins=num_sub_bins,
    )
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    x_full = _stack_history(sub_counts, k_history=k_history, split_starts=split_starts)
    x_test = x_full[test_idx]
    n_test, S, N = x_test.shape
    H = snn.hidden_dim
    input_synops_per_bin = x_test.sum(axis=(1, 2)).astype(np.float64)

    # Forward through the trained LIF to count hidden spikes (no grad).
    # Use the model's own _encode against its trained _W weights so the
    # decay constant + threshold logic stays in one place.
    with torch.no_grad():
        xt = torch.from_numpy(x_test).to(dtype=torch.float32)
        z = snn._encode(xt, snn._W)   # [n_test, H], per-bin total hidden spikes
        hidden_spikes_per_bin = z.sum(dim=1).cpu().numpy().astype(np.float64)

    return {
        "n_test_bins": int(n_test),
        "seq_len": int(S),
        "input_synops_total": float(input_synops_per_bin.sum()),
        "hidden_spikes_total": float(hidden_spikes_per_bin.sum()),
        "readout_macs_total": float(n_test * H * 2),
        "synops_per_prediction_mean": float(
            (input_synops_per_bin + hidden_spikes_per_bin + H * 2).mean()
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path,
                   default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("results/cluster/pareto/pareto_energy_accuracy.json"))
    p.add_argument("--hidden-dims", type=int, nargs="+",
                   default=[32, 64, 128, 256, 512])
    p.add_argument("--k-histories", type=int, nargs="+",
                   default=[0, 2, 4, 8])
    p.add_argument("--event-budget", type=float, default=1.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.30)
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

    f = args.event_budget
    et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)

    rows: list[dict] = []
    for h in args.hidden_dims:
        for k in args.k_histories:
            for seed in args.seeds:
                set_global_seed(seed)
                logger.info("=" * 72)
                logger.info("hidden_dim=%d k_history=%d seed=%d", h, k, seed)
                logger.info("=" * 72)

                snn = TrainedLatencySNN(
                    num_neurons=num_neurons, hidden_dim=h, threshold=args.threshold,
                    bin_size_ms=bin_size_ms, k_history=k,
                    tau_ms=args.tau_ms, epochs=args.epochs, patience=args.patience,
                    seed=seed,
                ).fit(et, en, y, train_idx, val_idx)
                split_starts = (int(train_idx.min()), int(val_idx.min()),
                                int(test_idx.min()))
                y_pred = snn.predict(et, en, test_idx, split_starts=split_starts)
                r2 = velocity_r2(y[test_idx], y_pred)

                num_sub_bins = snn.num_sub_bins
                ops = _count_synops(
                    snn, et, en, test_idx, train_idx, val_idx,
                    num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                    num_sub_bins=num_sub_bins, k_history=k,
                )

                energy_pj = {
                    chip: ops["synops_per_prediction_mean"] * pj
                    for chip, pj in MAC_ENERGY_PJ.items()
                }

                row = {
                    "hidden_dim": int(h),
                    "k_history": int(k),
                    "seed": int(seed),
                    "event_budget": float(f),
                    "r2_joint": float(r2["r2_joint"]),
                    "r2_vx": float(r2["r2_vx"]),
                    "r2_vy": float(r2["r2_vy"]),
                    "best_val_r2": float(snn.best_val_r2),
                    "synops_per_prediction_mean": ops["synops_per_prediction_mean"],
                    "input_synops_total": ops["input_synops_total"],
                    "hidden_spikes_total": ops["hidden_spikes_total"],
                    "readout_macs_total": ops["readout_macs_total"],
                    "energy_pj_per_prediction": energy_pj,
                }
                rows.append(row)
                logger.info(
                    "h=%d k=%d seed=%d  r2=%+.4f  synops/pred=%.0f  loihi2=%.1f pJ",
                    h, k, seed, row["r2_joint"], row["synops_per_prediction_mean"],
                    energy_pj.get("loihi2", float("nan")),
                )

                # Stream after every config so partial results survive crashes.
                args.out.write_text(json.dumps({"rows": rows}, indent=2))

    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    logger.info("wrote %s (%d rows)", args.out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
