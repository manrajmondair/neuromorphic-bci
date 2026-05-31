"""True online streaming simulation — feed test bins one at a time, decode causally.

For each model with a fitted decoder, we simulate a deployed BCI:

  1. Walk through test_idx in time order.
  2. At each bin, truncate the bin's events to the first `window_ms`
     (the causal lookahead a real BCI has).
  3. Run the decoder on this single bin (or its bin + history if the
     model takes history features), record the prediction time.
  4. Integrate the predicted velocity to a cumulative cursor position.
  5. Compare against the integrated true velocity for per-bin error
     and cumulative drift.

For the trained SNN this runs the encoder one bin at a time so we can
also measure per-bin wall-clock latency (T3c). The ridge models run on
a single feature vector per bin so latency is trivially fast.

Writes:
  * results/streaming/{model}_streaming.json  -- per-bin latencies + R^2 + drift
  * results/figures/streaming_drift.png       -- cumulative drift over time
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocess import load_processed
from src.evaluation.figstyle import apply_style, color_for, label_for, panel, save_fig
from src.evaluation.metrics import velocity_r2
from src.features.causal_window import truncate_to_window
from src.features.spike_counts import counts_from_events, stack_lag_features
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_online_streaming")


def _fit_ridge(spike_counts, velocity, train_idx, val_idx, lag_bins, split_starts):
    X = stack_lag_features(
        spike_counts.astype(np.float32), num_lags=lag_bins, split_starts=split_starts,
    )
    decoder = RidgeDecoder(alphas=DEFAULT_ALPHAS).fit(
        X[train_idx], velocity[train_idx], X[val_idx], velocity[val_idx],
    )
    return decoder, X


def _stream_ridge(decoder, X, test_idx):
    """Per-bin ridge inference; record wall-clock latency per call."""
    preds = np.zeros((test_idx.size, 2), dtype=np.float32)
    latencies = np.zeros(test_idx.size, dtype=np.float64)
    for k, t in enumerate(test_idx):
        x_t = X[t : t + 1]
        start = time.perf_counter()
        preds[k] = decoder.predict(x_t)[0]
        latencies[k] = time.perf_counter() - start
    return preds, latencies


def _stream_trained_snn(snn, et, en, test_idx, split_starts):
    """Per-bin trained-SNN inference with pre-computed sub-bin tensor.

    The history stack is computed *once* before the timing loop (it is
    not part of the per-bin inference cost in a deployed BCI — the
    history buffer is maintained incrementally). Per-bin latency
    measures only the LIF forward pass + readout on a single
    [1, S_total, N] tensor.
    """
    import torch

    from src.models.trained_snn import (
        _sparse_events_to_subbin_counts,
        _stack_history,
    )

    x_all = _sparse_events_to_subbin_counts(
        et, en, snn.num_neurons, snn.bin_size_ms, snn.num_sub_bins,
    )
    x_all = _stack_history(x_all, snn.k_history, split_starts)

    W = snn._W
    W_out = snn._W_out
    b_out = snn._b_out

    preds = np.zeros((test_idx.size, 2), dtype=np.float32)
    latencies = np.zeros(test_idx.size, dtype=np.float64)
    # The trained model's weights may live on CUDA after the CUDA-aware fit()
    # patch — keep the per-bin input tensor on whatever device W lives on.
    device = W.device
    with torch.no_grad():
        for k, t in enumerate(test_idx):
            x_t = torch.from_numpy(x_all[t : t + 1]).to(device)
            start = time.perf_counter()
            z = snn._encode(x_t, W)
            y_pred = z @ W_out.T + b_out
            latencies[k] = time.perf_counter() - start
            preds[k] = y_pred.cpu().numpy()[0]
    return preds, latencies


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/streaming"))
    p.add_argument("--fig-path", type=Path, default=Path("results/figures/supp_streaming_drift.png"))
    p.add_argument("--window-ms", type=float, default=50.0,
                   help="causal lookahead per bin (defaults to full bin)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snn-hidden-dim", type=int, default=128)
    p.add_argument("--snn-k-history", type=int, default=4)
    p.add_argument("--snn-epochs", type=int, default=80)
    p.add_argument("--snn-patience", type=int, default=15)
    p.add_argument("--snn-threshold", type=float, default=0.30)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    bin_size_ms = int(data["bin_size_ms"])

    # Truncate every bin to the causal window once (offline shortcut — the
    # real online decoder would do this per bin too).
    et, en = truncate_to_window(
        data["event_times"], data["event_neurons"], window_ms=args.window_ms,
    )
    spike_counts = counts_from_events(en, int(data["num_neurons"])).astype(np.float32)

    set_global_seed(args.seed)
    streams = {}

    # Ridge single-bin
    logger.info("fitting ridge (lag=0) for streaming ...")
    dec_r, X_r = _fit_ridge(spike_counts, y, train_idx, val_idx, lag_bins=0,
                            split_starts=split_starts)
    preds, lat = _stream_ridge(dec_r, X_r, test_idx)
    streams["ridge"] = (preds, lat)

    # Ridge lag-4
    logger.info("fitting ridge_lag4 for streaming ...")
    dec_r4, X_r4 = _fit_ridge(spike_counts, y, train_idx, val_idx, lag_bins=4,
                              split_starts=split_starts)
    preds, lat = _stream_ridge(dec_r4, X_r4, test_idx)
    streams["ridge_lag4"] = (preds, lat)

    # Trained SNN
    logger.info("fitting trained_snn for streaming ...")
    snn = TrainedLatencySNN(
        num_neurons=int(data["num_neurons"]),
        hidden_dim=args.snn_hidden_dim,
        threshold=args.snn_threshold,
        bin_size_ms=bin_size_ms,
        k_history=args.snn_k_history,
        epochs=args.snn_epochs,
        patience=args.snn_patience,
        seed=args.seed,
    ).fit(et, en, y, train_idx, val_idx)
    preds, lat = _stream_trained_snn(snn, et, en, test_idx, split_starts)
    streams["trained_snn"] = (preds, lat)

    # Per-model analysis
    bin_size_s = bin_size_ms / 1000.0
    true_pos = np.cumsum(y[test_idx] * bin_size_s, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows: list[dict] = []
    for model, (preds, lat) in streams.items():
        r2 = velocity_r2(y[test_idx], preds)
        pred_pos = np.cumsum(preds * bin_size_s, axis=0)
        drift = np.linalg.norm(pred_pos - true_pos, axis=1)
        cum_drift_at_t = drift  # already cumulative through cumsum
        row = {
            "model": model,
            "window_ms": float(args.window_ms),
            "n_test_bins": int(test_idx.size),
            "r2_joint": float(r2["r2_joint"]),
            "r2_vx": float(r2["r2_vx"]),
            "r2_vy": float(r2["r2_vy"]),
            "mean_latency_ms": float(lat.mean() * 1000),
            "median_latency_ms": float(np.median(lat) * 1000),
            "p95_latency_ms": float(np.percentile(lat, 95) * 1000),
            "p99_latency_ms": float(np.percentile(lat, 99) * 1000),
            "final_drift": float(drift[-1]),
            "mean_drift": float(drift.mean()),
        }
        rows.append(row)
        (args.out_dir / f"{model}_streaming.json").write_text(json.dumps(row, indent=2))
        logger.info(
            "%-12s r2=%+.4f  lat: mean=%.3f ms median=%.3f ms p95=%.3f ms  drift_final=%.2f",
            model, row["r2_joint"], row["mean_latency_ms"], row["median_latency_ms"],
            row["p95_latency_ms"], row["final_drift"],
        )
        t_axis = np.arange(preds.shape[0]) * bin_size_s
        c = color_for(model)
        axes[0].plot(t_axis, drift, color=c, label=f"{label_for(model)}")
        axes[1].hist(lat * 1000, bins=40, alpha=0.55, color=c, label=label_for(model))

    axes[0].set_xlabel("Time  (s)")
    axes[0].set_ylabel("Cumulative position drift  (mm)")
    axes[0].legend(loc="upper left")
    panel(axes[0], "a")

    axes[1].set_xlabel("Per-bin inference latency  (ms)")
    axes[1].set_ylabel("Bin count")
    axes[1].legend(loc="upper right")
    panel(axes[1], "b")
    fig.tight_layout()
    save_fig(fig, args.fig_path.with_suffix(""))
    logger.info("wrote %s", args.fig_path)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({"rows": rows, "window_ms": args.window_ms}, indent=2))
    logger.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
