"""Analytical Loihi 2 deployment estimate for the trained SNN.

Intel's `lava-nc` framework is not pip-installable on macOS ARM, so a
direct cycle-accurate simulation is not available in this environment.
Instead we fit the trained SNN, measure its *actual* per-bin input
synaptic-operation count + hidden spike-count on the test split, and
multiply by published per-SynOp energy figures:

  * Loihi 2 — Hala Point family, ~23 pJ/SynOp at full activity
    (Mayr & Davies, 2024 press materials).
  * NorthPole INT8 — ~2 pJ/MAC equivalent (Modha et al., Science 2023).

For each test bin we count:
  * `input_synops`   = number of input spikes processed (events kept under
                       the current event budget).
  * `hidden_synops`  = number of hidden spikes emitted by the LIF stack.

Energy per prediction = (input_synops + hidden_synops + readout_macs) *
                        per_synop_energy.

This is an upper bound on what the chip would consume because we count
every input event as a separate SynOp (Loihi's compressed synaptic
fanout would typically lower this slightly).

Writes results/loihi/loihi_estimate.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.efficiency_tracker import MAC_ENERGY_PJ
from src.features.event_budget import apply_event_budget
from src.models.trained_snn import (
    TrainedLatencySNN,
    _sparse_events_to_subbin_counts,
    _stack_history,
)
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_loihi_estimate")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out", type=Path, default=Path("results/loihi/loihi_estimate.json"))
    p.add_argument("--event-budgets", type=float, nargs="+", default=[1.00, 0.50, 0.25, 0.10])
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])
    bin_size_s = bin_size_ms / 1000.0
    n_test = test_idx.size

    rows = []
    for f in args.event_budgets:
        logger.info("=" * 72)
        logger.info("loihi estimate: f=%.2f", f)
        logger.info("=" * 72)
        et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
        set_global_seed(args.seed)
        snn = TrainedLatencySNN(
            num_neurons=num_neurons, hidden_dim=args.hidden_dim,
            threshold=args.threshold, bin_size_ms=bin_size_ms,
            k_history=args.k_history, epochs=args.epochs, patience=args.patience,
            seed=args.seed,
        ).fit(et, en, y, train_idx, val_idx)

        # Replay the test bins and measure actual SynOp/spike counts.
        x_all = _sparse_events_to_subbin_counts(
            et, en, num_neurons=num_neurons,
            bin_size_ms=bin_size_ms, num_sub_bins=snn.num_sub_bins,
        )
        split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
        x_all = _stack_history(x_all, args.k_history, split_starts)
        x_test = x_all[test_idx]
        # Input SynOps per test bin = total input spike count seen by the
        # LIF (including history-padded sub-bins, since the LIF must
        # process every sub-bin's pulse stream).
        input_synops_per_bin = x_test.sum(axis=(1, 2)).astype(np.float64)  # [n_test]
        # Hidden SynOps = total spikes emitted from the hidden layer.
        # Run encoder once to count emitted spikes (this is z without grad).
        import torch
        with torch.no_grad():
            # snn._W is on whatever device fit() left it on — keep xt aligned.
            device = snn._W.device
            xt = torch.from_numpy(x_test).to(device)
            z = snn._encode(xt, snn._W).cpu().numpy()
        hidden_spikes_per_bin = z.sum(axis=1).astype(np.float64)
        # Readout MACs = 2 * hidden_dim per bin (constant).
        readout_macs_per_bin = 2 * args.hidden_dim

        total_input = float(input_synops_per_bin.sum())
        total_hidden = float(hidden_spikes_per_bin.sum())
        total_readout = float(readout_macs_per_bin * n_test)
        total_synops = total_input + total_hidden + total_readout

        per_chip = {}
        for chip, pj_per_op in MAC_ENERGY_PJ.items():
            total_pj = total_synops * pj_per_op
            per_chip[chip] = {
                "pj_per_op": float(pj_per_op),
                "energy_total_uj": float(total_pj) / 1e6,
                "energy_per_prediction_uj": float(total_pj) / 1e6 / n_test,
                "avg_power_uw": float(total_pj) / 1e6 / (n_test * bin_size_s),
            }
        logger.info(
            "budget f=%.2f best_val_r2=%+.4f  test SynOps: input=%.0f hidden=%.0f readout=%.0f",
            f, snn.best_val_r2, total_input, total_hidden, total_readout,
        )
        for chip, ent in per_chip.items():
            logger.info(
                "  %s: %.4f µJ/prediction, %.1f µW average",
                chip, ent["energy_per_prediction_uj"], ent["avg_power_uw"],
            )

        rows.append({
            "event_budget": float(f),
            "n_test_bins": int(n_test),
            "best_val_r2": float(snn.best_val_r2),
            "input_synops_total": total_input,
            "hidden_spikes_total": total_hidden,
            "readout_macs_total": total_readout,
            "synops_total": total_synops,
            "synops_per_prediction_mean": float(input_synops_per_bin.mean()
                                                + hidden_spikes_per_bin.mean()
                                                + readout_macs_per_bin),
            "per_chip": per_chip,
        })

    out_blob = {
        "model": "trained_snn",
        "hidden_dim": int(args.hidden_dim),
        "k_history": int(args.k_history),
        "bin_size_ms": bin_size_ms,
        "mac_energy_pj": dict(MAC_ENERGY_PJ),
        "rows": rows,
        "notes": (
            "lava-nc / NxSDK has no pip wheel on macOS ARM. Numbers here use "
            "the trained SNN's measured SynOp counts on the held-out test "
            "split, multiplied by published per-SynOp energy figures. "
            "Real Loihi 2 silicon would likely be slightly lower because "
            "fanout-compressed connections reduce SynOp counts; this is "
            "an upper bound."
        ),
    }
    args.out.write_text(json.dumps(out_blob, indent=2))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
