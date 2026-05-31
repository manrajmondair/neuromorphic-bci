"""Generate the reproducible history-depth sweeps behind the context figures.

Writes:
  results/sweeps/history_depth.csv       model, k_bins, seed, r2_joint
      R² vs readout history depth for ridge, reservoir SNN, trained SNN — the
      "temporal context is the dominant lever" story (all converge as k grows).
  results/sweeps/trained_snn_context.csv  mechanism, k_bins, seed, r2_joint
      Trained SNN with history on the *input* (k_history; saturates past ~8
      bins) vs on the *readout* (readout_lag; keeps climbing to ~0.68).

f = 1.0 throughout (the context question is cleanest at full budget). CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import Ridge

from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.features.spike_counts import stack_lag_features
from src.models.snn_decoder import SparseLatencySNN
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(message)s")
logger = logging.getLogger("history_sweeps")
ALPHAS = [1e1, 1e2, 1e3, 3e3, 1e4, 3e4, 1e5]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/sweeps"))
    p.add_argument("--depths", type=int, nargs="+", default=[0, 2, 4, 8, 12, 16, 20, 24, 28])
    p.add_argument("--input-depths", type=int, nargs="+", default=[0, 2, 4, 8, 12, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--tau-ms", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=120)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    tr, va, te = data["train_idx"], data["val_idx"], data["test_idx"]
    y = np.asarray(data["velocity"], np.float32)
    NN, BS = int(data["num_neurons"]), int(data["bin_size_ms"])
    ss = (int(tr.min()), int(va.min()), int(te.min()))
    et, en = apply_event_budget(data["event_times"], data["event_neurons"], 1.0)
    counts = np.asarray(data["spike_counts"], np.float32)

    def ridge_best(F):
        b = -np.inf; bt = None
        for a in ALPHAS:
            r = Ridge(alpha=a).fit(F[tr], y[tr])
            v = velocity_r2(y[va], r.predict(F[va]))["r2_joint"]
            if v > b:
                b = v; bt = velocity_r2(y[te], r.predict(F[te]))["r2_joint"]
        return bt

    # ---- history_depth.csv: ridge, reservoir SNN, trained SNN vs depth ----
    rows = []
    # reservoir encodings cached per seed (encode once, restack per depth)
    for seed in args.seeds:
        m = SparseLatencySNN(num_neurons=NN, hidden_dim=args.hidden_dim, tau_ms=args.tau_ms,
                             threshold=0.05, bin_size_ms=BS, n_restarts=1, seed=seed)
        S = m._encode_with_W(m._init_W(NN, args.hidden_dim, seed), et, en)
        mu = S[tr].mean(0); sg = S[tr].std(0) + 1e-6
        Sz = ((S - mu) / sg).astype(np.float32)
        for k in args.depths:
            # ridge on raw counts (seed-invariant, compute on seed 0 only)
            if seed == args.seeds[0]:
                rt = ridge_best(stack_lag_features(counts, k, ss))
                rows.append({"model": "ridge_hist" if k > 0 else "ridge",
                             "k_bins": k, "seed": seed, "r2_joint": rt})
            res_t = ridge_best(stack_lag_features(Sz, k, ss))
            rows.append({"model": "reservoir_snn", "k_bins": k, "seed": seed, "r2_joint": res_t})
            logger.info("history_depth seed=%d k=%d  reservoir=%+.4f", seed, k, res_t)
    # trained SNN (readout-lag) vs depth
    for seed in args.seeds:
        for k in args.depths:
            set_global_seed(seed)
            snn = TrainedLatencySNN(num_neurons=NN, hidden_dim=args.hidden_dim, tau_ms=args.tau_ms,
                                    threshold=0.30, bin_size_ms=BS, readout_lag=k,
                                    epochs=args.epochs, patience=20, seed=seed).fit(et, en, y, tr, va)
            t = velocity_r2(y[te], snn.predict(et, en, te, split_starts=ss))["r2_joint"]
            rows.append({"model": "trained_snn", "k_bins": k, "seed": seed, "r2_joint": t})
            logger.info("history_depth seed=%d k=%d  trained_snn=%+.4f", seed, k, t)
    with (args.out_dir / "history_depth.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "k_bins", "seed", "r2_joint"]); w.writeheader(); w.writerows(rows)
    logger.info("wrote %s (%d rows)", args.out_dir / "history_depth.csv", len(rows))

    # ---- trained_snn_context.csv: input-history vs readout-lag ----
    # readout-lag is identical to the trained_snn curve above — reuse it.
    crows = [{"mechanism": "readout_lag", "k_bins": r["k_bins"], "seed": r["seed"],
              "r2_joint": r["r2_joint"]} for r in rows if r["model"] == "trained_snn"]
    for seed in args.seeds:
        for k in args.input_depths:  # input-history (slower; LIF sees (k+1)*sub_bins steps)
            set_global_seed(seed)
            snn = TrainedLatencySNN(num_neurons=NN, hidden_dim=args.hidden_dim, tau_ms=args.tau_ms,
                                    threshold=0.30, bin_size_ms=BS, k_history=k,
                                    epochs=args.epochs, patience=20, seed=seed).fit(et, en, y, tr, va)
            t = velocity_r2(y[te], snn.predict(et, en, te, split_starts=ss))["r2_joint"]
            crows.append({"mechanism": "input_history", "k_bins": k, "seed": seed, "r2_joint": t})
            logger.info("context seed=%d k=%d input_history=%+.4f", seed, k, t)
    with (args.out_dir / "trained_snn_context.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mechanism", "k_bins", "seed", "r2_joint"]); w.writeheader(); w.writerows(crows)
    logger.info("wrote %s (%d rows)", args.out_dir / "trained_snn_context.csv", len(crows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
