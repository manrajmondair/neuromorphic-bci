"""Decoder analysis — what did the trained SNN learn?

For the canonical (f=1.0, k_history=4, seed=0) configuration:

  1. Fit a TrainedLatencySNN and extract the input projection W
     (shape [hidden_dim, num_neurons]) and the readout W_out
     ([2, hidden_dim]).

  2. Compute each *input neuron's* preferred direction (PD) from the
     spike-triggered velocity average:
         PD_n = atan2( sum_t count_n[t] * v_y[t],
                       sum_t count_n[t] * v_x[t] )
     and depth = || (sum count * v_x, sum count * v_y) || / sum count.

  3. Compute each *hidden unit's* readout direction:
         W_out[:, h] -> (cos theta_h, sin theta_h) -> theta_h
     This is the direction of cursor velocity that hidden unit h
     contributes to when it fires.

  4. Plot:
     a) Heatmap of W (rows = hidden units sorted by readout PD,
        columns = input neurons sorted by their tuning PD).
     b) Polar histogram of input-neuron PDs (M1 tuning sanity check).
     c) Polar histogram of hidden-unit readout PDs.
     d) Scatter of input PD vs the sum of W rows weighted by hidden PD
        — quantifies how aligned the SNN's projection is with biology.

Writes:
  * results/figures/trained_snn_weight_heatmap.png
  * results/figures/trained_snn_pref_directions.png
  * results/snn_weights/analysis.json (numbers + per-unit table)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocess import load_processed
from src.features.event_budget import apply_event_budget
from src.features.spike_counts import counts_from_events
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("analyze_trained_snn_weights")


def _input_preferred_directions(spike_counts: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each input neuron, compute (preferred_direction_rad, tuning_depth).

    spike_counts: [num_bins, num_neurons]
    velocity:     [num_bins, 2]  (vx, vy)
    """
    counts = spike_counts.astype(np.float64)
    total_counts = counts.sum(axis=0) + 1e-12  # [num_neurons]
    # Weighted sum of velocity by each neuron's firing.
    vx_w = (counts * velocity[:, 0:1]).sum(axis=0) / total_counts
    vy_w = (counts * velocity[:, 1:2]).sum(axis=0) / total_counts
    pd = np.arctan2(vy_w, vx_w)
    depth = np.sqrt(vx_w ** 2 + vy_w ** 2)
    return pd.astype(np.float32), depth.astype(np.float32)


def _hidden_preferred_directions(W_out: np.ndarray) -> np.ndarray:
    """For each hidden unit h, theta_h = atan2(W_out[1, h], W_out[0, h])."""
    return np.arctan2(W_out[1], W_out[0])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-fig-heatmap", type=Path,
                   default=Path("results/figures/trained_snn_weight_heatmap.png"))
    p.add_argument("--out-fig-pd", type=Path,
                   default=Path("results/figures/trained_snn_pref_directions.png"))
    p.add_argument("--out-analysis", type=Path,
                   default=Path("results/snn_weights/analysis.json"))
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_fig_heatmap.parent.mkdir(parents=True, exist_ok=True)
    args.out_analysis.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    set_global_seed(args.seed)
    # Use full event budget for the canonical analysis.
    et, en = apply_event_budget(data["event_times"], data["event_neurons"], 1.00)

    snn = TrainedLatencySNN(
        num_neurons=int(data["num_neurons"]),
        hidden_dim=args.hidden_dim,
        threshold=args.threshold,
        bin_size_ms=int(data["bin_size_ms"]),
        k_history=args.k_history,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    ).fit(et, en, y, train_idx, val_idx)
    logger.info("trained SNN: best_val_r2=%+.4f epochs=%d hidden_dim=%d k_history=%d",
                snn.best_val_r2, len(snn.history), args.hidden_dim, args.k_history)

    W = snn.W            # [hidden, num_neurons]
    W_out = snn.W_out    # [2, hidden]
    spike_counts = counts_from_events(en, int(data["num_neurons"]))

    # Use train + val velocity for PD estimation (test is held out).
    fit_mask = np.concatenate([train_idx, val_idx])
    input_pd, input_depth = _input_preferred_directions(
        spike_counts[fit_mask], y[fit_mask]
    )
    hidden_pd = _hidden_preferred_directions(W_out)

    # --- figure 1: weight heatmap (sorted by PD) ---
    in_order = np.argsort(input_pd)
    hid_order = np.argsort(hidden_pd)
    W_sorted = W[np.ix_(hid_order, in_order)]
    vmax = float(np.percentile(np.abs(W_sorted), 99))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(W_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel(f"Input neuron (sorted by preferred direction, N={W.shape[1]})")
    ax.set_ylabel(f"Hidden unit (sorted by readout direction, H={W.shape[0]})")
    ax.set_title("Trained SNN input projection W (rows + cols sorted by preferred direction)")
    fig.colorbar(im, ax=ax, label="weight (signed)")
    fig.tight_layout()
    fig.savefig(args.out_fig_heatmap, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.out_fig_heatmap)

    # --- figure 2: PD polar histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), subplot_kw={"projection": "polar"})
    n_bins = 24
    axes[0].hist(input_pd, bins=n_bins, weights=input_depth, alpha=0.85,
                 color="#1f77b4", edgecolor="white")
    axes[0].set_title(f"Input neurons (N={W.shape[1]})\nweighted by tuning depth")
    axes[1].hist(hidden_pd, bins=n_bins, alpha=0.85, color="#ff7f0e", edgecolor="white")
    axes[1].set_title(f"Hidden units (H={W.shape[0]})\nW_out direction")
    fig.suptitle("Preferred-direction distributions: input neurons vs trained SNN hidden units",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_fig_pd, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.out_fig_pd)

    # --- numeric summary ---
    # For each hidden unit, project its input-row onto the input PDs:
    # if the trained encoder is biology-aligned, hidden unit h should
    # weight input neurons with similar PDs more strongly. Quantify via
    # the cosine of the difference between each hidden unit's readout
    # direction and a "ranking" of input PDs weighted by |W[h]|.
    coh = np.zeros(W.shape[0], dtype=np.float32)
    for h in range(W.shape[0]):
        w_h = np.abs(W[h])  # take magnitudes
        w_h = w_h / (w_h.sum() + 1e-12)
        # Vector mean direction weighted by |W[h]|
        cx = (w_h * np.cos(input_pd)).sum()
        cy = (w_h * np.sin(input_pd)).sum()
        # angle of this mean
        mean_dir = float(np.arctan2(cy, cx))
        coh[h] = float(np.cos(mean_dir - hidden_pd[h]))

    analysis = {
        "config": {
            "hidden_dim": int(args.hidden_dim),
            "k_history": int(args.k_history),
            "threshold": float(args.threshold),
            "seed": int(args.seed),
            "best_val_r2": float(snn.best_val_r2),
        },
        "input_neuron_preferred_directions_rad": input_pd.tolist(),
        "input_neuron_tuning_depths": input_depth.tolist(),
        "hidden_unit_readout_directions_rad": hidden_pd.tolist(),
        "hidden_unit_coherence": coh.tolist(),
        "summary": {
            "mean_input_tuning_depth": float(input_depth.mean()),
            "median_input_tuning_depth": float(np.median(input_depth)),
            "hidden_unit_mean_coherence": float(coh.mean()),
            "hidden_unit_aligned_fraction": float((coh > 0.5).mean()),
        },
    }
    args.out_analysis.write_text(json.dumps(analysis, indent=2))
    logger.info(
        "wrote %s  mean_coherence=%+.4f aligned_fraction=%.2f",
        args.out_analysis,
        analysis["summary"]["hidden_unit_mean_coherence"],
        analysis["summary"]["hidden_unit_aligned_fraction"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
