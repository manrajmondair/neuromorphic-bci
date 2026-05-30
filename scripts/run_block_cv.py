"""Leave-one-block-out temporal cross-validation — within-recording stand-in for cross-session.

MC_RTT has a single behaviour-labeled recording (sub-Indy_desc-train),
so true cross-session generalization within the dandiset is not
available. We approximate it with a 4-fold leave-one-block-out CV:

  1. Drop the canonical boundary gap.
  2. Slice all usable bins into 4 disjoint, time-contiguous quarters.
  3. For each fold q:
       train_idx = bins outside Q
       val_idx   = first half of Q
       test_idx  = second half of Q
  4. Fit ridge (lag 0 and lag 4) and the trained SNN, score on the
     fold's test_idx, record R^2 + bootstrap CI.

Reporting R^2 vs fold index sweeps the entire recording, so we can
detect any temporal regime where the decoder breaks (a stronger
chronic-drift probe than the existing early/late split).

Writes results/block_cv/{model}_block_cv.json and a summary plot.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge

from src.data.preprocess import load_processed
from src.evaluation.metrics import velocity_r2, velocity_r2_bootstrap
from src.features.event_budget import restrict_to_event_budget
from src.features.spike_counts import counts_from_events, stack_lag_features
from src.models.deeper_snn import DeeperTrainedSNN
from src.models.ridge_decoder import DEFAULT_ALPHAS, RidgeDecoder
from src.models.snn_decoder import SparseLatencySNN
from src.models.trained_snn import TrainedLatencySNN
from src.utils.seed import set_global_seed

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_block_cv")


def make_folds(num_bins: int, n_folds: int = 4, boundary_gap: int = 1):
    bins = np.arange(num_bins)
    edges = np.linspace(0, num_bins, n_folds + 1).astype(int)
    folds = []
    for q in range(n_folds):
        q_lo, q_hi = edges[q], edges[q + 1]
        # Within the fold: first half = val, second half = test
        q_mid = (q_lo + q_hi) // 2
        val_idx = np.arange(q_lo + boundary_gap, q_mid - boundary_gap)
        test_idx = np.arange(q_mid + boundary_gap, q_hi - boundary_gap)
        # Training: everything outside the fold, minus boundary gaps.
        train_mask = np.ones(num_bins, dtype=bool)
        train_mask[max(0, q_lo - boundary_gap) : min(num_bins, q_hi + boundary_gap)] = False
        train_idx = bins[train_mask]
        folds.append({"fold": q, "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx})
    return folds


def _fit_ridge(spike_counts, velocity, train_idx, val_idx, test_idx, lag_bins):
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    X = stack_lag_features(spike_counts.astype(np.float32), num_lags=lag_bins,
                           split_starts=split_starts)
    decoder = RidgeDecoder(alphas=DEFAULT_ALPHAS).fit(
        X[train_idx], velocity[train_idx], X[val_idx], velocity[val_idx],
    )
    return decoder.predict(X[test_idx]), decoder.best_alpha


def _fit_snn(et, en, velocity, train_idx, val_idx, test_idx,
             num_neurons, bin_size_ms, hidden_dim, k_history,
             threshold, epochs, patience, seed):
    snn = TrainedLatencySNN(
        num_neurons=num_neurons, hidden_dim=hidden_dim, threshold=threshold,
        bin_size_ms=bin_size_ms, k_history=k_history,
        epochs=epochs, patience=patience, seed=seed,
    ).fit(et, en, velocity, train_idx, val_idx)
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    return snn.predict(et, en, test_idx, split_starts=split_starts), snn.best_val_r2


def _fit_reservoir_snn(et, en, velocity, train_idx, val_idx, test_idx,
                       num_neurons, bin_size_ms, hidden_dim, tau_ms, lag_bins,
                       thresholds, alphas, seed):
    """Fixed random-projection LIF reservoir + lag-stacked ridge readout.

    Numpy-only (no training of the encoder). Encodes once per threshold,
    builds the boundary-safe history window over the hidden activity, and
    selects (threshold, ridge alpha) on the fold's val split. Matched to
    ridge_lag4 / the trained SNN at a k=4 (200 ms) history window so the
    block-CV comparison is apples-to-apples on temporal context.
    """
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    base = SparseLatencySNN(num_neurons=num_neurons, hidden_dim=hidden_dim,
                            tau_ms=tau_ms, threshold=float(thresholds[0]),
                            bin_size_ms=bin_size_ms, n_restarts=1, seed=seed)
    W = base._init_W(num_neurons, hidden_dim, seed)
    best = None  # (val_r2, test_pred, threshold, alpha)
    for thr in thresholds:
        base.threshold = float(thr)
        S = base._encode_with_W(W, et, en)
        mu = S[train_idx].mean(axis=0)
        sigma = S[train_idx].std(axis=0) + 1e-6
        F = stack_lag_features(((S - mu) / sigma).astype(np.float32), num_lags=lag_bins,
                               split_starts=split_starts)
        for a in alphas:
            ro = Ridge(alpha=float(a)).fit(F[train_idx], velocity[train_idx])
            vr = velocity_r2(velocity[val_idx], ro.predict(F[val_idx]))["r2_joint"]
            if best is None or vr > best[0]:
                best = (vr, ro.predict(F[test_idx]), float(thr), float(a))
    return best[1], best[2], best[3]


def _fit_deeper_snn(et, en, velocity, train_idx, val_idx, test_idx,
                    num_neurons, bin_size_ms, hidden_dims, readout_lag, tau_ms,
                    threshold, epochs, patience, seed):
    """Multi-layer LIF with a lag-stacked readout, fit per fold at a matched
    history window (readout_lag bins). Feedforward — recurrence destabilises."""
    from src.utils.seed import set_global_seed
    set_global_seed(seed)
    snn = DeeperTrainedSNN(
        num_neurons=num_neurons, hidden_dims=tuple(hidden_dims), tau_ms=tau_ms,
        threshold=threshold, bin_size_ms=bin_size_ms, readout_lag=readout_lag,
        recurrent=False, epochs=epochs, patience=patience, seed=seed,
    ).fit(et, en, velocity, train_idx, val_idx)
    split_starts = (int(train_idx.min()), int(val_idx.min()), int(test_idx.min()))
    return snn.predict(et, en, test_idx, split_starts=split_starts), snn.best_val_r2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-path", type=Path, default=Path("data/processed/processed_mc_rtt.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/block_cv"))
    p.add_argument("--fig-path", type=Path, default=Path("results/figures/block_cv.png"))
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--boundary-gap", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument(
        "--event-budgets",
        type=float,
        nargs="+",
        default=[1.0],
        help="event budgets to sweep across (default keeps all events)",
    )
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--k-history", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--skip-snn", action="store_true", help="skip the trained (BPTT) SNN")
    p.add_argument("--skip-ridge", action="store_true", help="skip the ridge baselines")
    p.add_argument("--reservoir-snn", action="store_true",
                   help="also fit the numpy reservoir SNN (random-proj LIF + lag-stacked ridge)")
    p.add_argument("--reservoir-hidden-dim", type=int, default=1024)
    p.add_argument("--reservoir-tau-ms", type=float, default=10.0)
    p.add_argument("--reservoir-lag-bins", type=int, default=4,
                   help="reservoir readout history depth (default 4 ≈ 200 ms, matched to ridge_lag4)")
    p.add_argument("--reservoir-thresholds", type=float, nargs="+", default=[0.05, 0.20])
    p.add_argument("--reservoir-alphas", type=float, nargs="+", default=[1e3, 1e4, 3e4, 1e5])
    p.add_argument("--deeper-snn", action="store_true",
                   help="also fit the multi-layer (deeper) LIF SNN with a lag-stacked readout")
    p.add_argument("--deeper-hidden-dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--deeper-readout-lag", type=int, default=4,
                   help="deeper-SNN readout history depth (default 4 ≈ 200 ms, matched window)")
    p.add_argument("--deeper-tau-ms", type=float, default=10.0)
    p.add_argument("--deeper-threshold", type=float, default=0.30)
    p.add_argument("--deeper-epochs", type=int, default=100)
    p.add_argument("--merge-into", type=Path, default=None,
                   help="merge the computed rows into an existing block_cv.json (replacing rows "
                        "for the same models) instead of writing a fresh file")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_processed(args.processed_path)
    y = np.asarray(data["velocity"], dtype=np.float32)
    num_bins = int(data["spike_counts"].shape[0])
    num_neurons = int(data["num_neurons"])
    bin_size_ms = int(data["bin_size_ms"])

    folds = make_folds(num_bins, n_folds=args.n_folds, boundary_gap=args.boundary_gap)

    # Cache budget-filtered event lists / spike-count matrices so the
    # inner loops don't recompute them per fold or seed.
    budget_cache: dict[float, tuple[np.ndarray, list, list]] = {}
    for f in args.event_budgets:
        sub = restrict_to_event_budget(data, fraction=f)
        budget_cache[float(f)] = (
            np.asarray(sub["spike_counts"], dtype=np.float32),
            sub["event_times"],
            sub["event_neurons"],
        )

    rows: list[dict] = []
    for f in args.event_budgets:
        spike_counts, et, en = budget_cache[float(f)]
        logger.info("#" * 72)
        logger.info("event budget f=%.2f  (events kept = %d)", f, sum(t.size for t in et))
        logger.info("#" * 72)
        for fold in folds:
            train_idx, val_idx, test_idx = fold["train_idx"], fold["val_idx"], fold["test_idx"]
            logger.info("=" * 72)
            logger.info("f=%.2f fold %d: train=%d val=%d test=%d (bins %d..%d)",
                        f, fold["fold"], train_idx.size, val_idx.size, test_idx.size,
                        int(test_idx.min()), int(test_idx.max()))
            logger.info("=" * 72)

            for seed in args.seeds:
                set_global_seed(seed)

                if not args.skip_ridge:
                    for lag in (0, 4):
                        y_pred, alpha = _fit_ridge(spike_counts, y, train_idx, val_idx, test_idx, lag)
                        r2 = velocity_r2(y[test_idx], y_pred)
                        r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
                        model = "ridge" if lag == 0 else "ridge_lag4"
                        rows.append({
                            "model": model, "event_budget": float(f),
                            "fold": int(fold["fold"]), "seed": int(seed),
                            "train_size": int(train_idx.size),
                            "test_size": int(test_idx.size),
                            "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                            "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                            "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                            "best_alpha": alpha, "n_boot": int(args.n_boot),
                        })
                        logger.info("%-11s f=%.2f fold=%d seed=%d  r2=%+.4f [%.4f, %.4f]",
                                    model, f, fold["fold"], seed, r2["r2_joint"],
                                    r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"])

                if args.reservoir_snn:
                    y_pred, r_thr, r_alpha = _fit_reservoir_snn(
                        et, en, y, train_idx, val_idx, test_idx,
                        num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                        hidden_dim=args.reservoir_hidden_dim, tau_ms=args.reservoir_tau_ms,
                        lag_bins=args.reservoir_lag_bins,
                        thresholds=args.reservoir_thresholds, alphas=args.reservoir_alphas,
                        seed=seed,
                    )
                    r2 = velocity_r2(y[test_idx], y_pred)
                    r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
                    rows.append({
                        "model": "reservoir_snn", "event_budget": float(f),
                        "fold": int(fold["fold"]), "seed": int(seed),
                        "train_size": int(train_idx.size),
                        "test_size": int(test_idx.size),
                        "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                        "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                        "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                        "tuned_threshold": r_thr, "best_alpha": r_alpha,
                        "lag_bins": int(args.reservoir_lag_bins), "n_boot": int(args.n_boot),
                    })
                    logger.info("reservoir_snn f=%.2f fold=%d seed=%d  r2=%+.4f [%.4f, %.4f] thr=%g a=%g",
                                f, fold["fold"], seed, r2["r2_joint"],
                                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"], r_thr, r_alpha)

                if args.deeper_snn:
                    y_pred, val_r2 = _fit_deeper_snn(
                        et, en, y, train_idx, val_idx, test_idx,
                        num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                        hidden_dims=args.deeper_hidden_dims, readout_lag=args.deeper_readout_lag,
                        tau_ms=args.deeper_tau_ms, threshold=args.deeper_threshold,
                        epochs=args.deeper_epochs, patience=20, seed=seed,
                    )
                    r2 = velocity_r2(y[test_idx], y_pred)
                    r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
                    rows.append({
                        "model": "deeper_snn", "event_budget": float(f),
                        "fold": int(fold["fold"]), "seed": int(seed),
                        "train_size": int(train_idx.size), "test_size": int(test_idx.size),
                        "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                        "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                        "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                        "best_val_r2": float(val_r2), "lag_bins": int(args.deeper_readout_lag),
                        "n_boot": int(args.n_boot),
                    })
                    logger.info("deeper_snn f=%.2f fold=%d seed=%d  r2=%+.4f [%.4f, %.4f] val=%+.4f",
                                f, fold["fold"], seed, r2["r2_joint"],
                                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"], val_r2)

                if not args.skip_snn:
                    y_pred, val_r2 = _fit_snn(
                        et, en, y, train_idx, val_idx, test_idx,
                        num_neurons=num_neurons, bin_size_ms=bin_size_ms,
                        hidden_dim=args.hidden_dim, k_history=args.k_history,
                        threshold=args.threshold, epochs=args.epochs,
                        patience=args.patience, seed=seed,
                    )
                    r2 = velocity_r2(y[test_idx], y_pred)
                    r2_boot = velocity_r2_bootstrap(y[test_idx], y_pred, n_boot=args.n_boot, seed=seed)
                    rows.append({
                        "model": "trained_snn", "event_budget": float(f),
                        "fold": int(fold["fold"]), "seed": int(seed),
                        "train_size": int(train_idx.size),
                        "test_size": int(test_idx.size),
                        "r2_vx": r2["r2_vx"], "r2_vy": r2["r2_vy"], "r2_joint": r2["r2_joint"],
                        "r2_joint_ci_lo": r2_boot["r2_joint_ci_lo"],
                        "r2_joint_ci_hi": r2_boot["r2_joint_ci_hi"],
                        "best_val_r2": float(val_r2),
                        "n_boot": int(args.n_boot),
                    })
                    logger.info("trained_snn f=%.2f fold=%d seed=%d  r2=%+.4f [%.4f, %.4f] val_r2=%+.4f",
                                f, fold["fold"], seed, r2["r2_joint"],
                                r2_boot["r2_joint_ci_lo"], r2_boot["r2_joint_ci_hi"], val_r2)

                # Stream progress to disk after every cell so partial-run JSONs are usable.
                (args.out_dir / "block_cv.json").write_text(json.dumps({"rows": rows}, indent=2))

    (args.out_dir / "block_cv.json").write_text(json.dumps({"rows": rows}, indent=2))

    if args.merge_into is not None:
        models_computed = {r["model"] for r in rows}
        existing = json.loads(args.merge_into.read_text())["rows"] if args.merge_into.exists() else []
        kept = [r for r in existing if r["model"] not in models_computed]
        merged = kept + rows
        args.merge_into.parent.mkdir(parents=True, exist_ok=True)
        args.merge_into.write_text(json.dumps({"rows": merged}, indent=2))
        logger.info("merged %d new rows (models=%s) into %s: kept %d existing -> %d total",
                    len(rows), sorted(models_computed), args.merge_into, len(kept), len(merged))

    # Summary table + plot.
    by_model_fold: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        by_model_fold.setdefault((r["model"], r["fold"]), []).append(r["r2_joint"])

    print()
    print(f"{'fold':>5} {'ridge':>10} {'ridge_lag4':>12} {'trained_snn':>12}")
    for q in range(args.n_folds):
        ridge_v = statistics.mean(by_model_fold.get(("ridge", q), [float("nan")]))
        ridge4_v = statistics.mean(by_model_fold.get(("ridge_lag4", q), [float("nan")]))
        snn_v = statistics.mean(by_model_fold.get(("trained_snn", q), [float("nan")]))
        print(f"{q:>5} {ridge_v:>+10.4f} {ridge4_v:>+12.4f} {snn_v:>+12.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    folds_axis = list(range(args.n_folds))
    for model in ("ridge", "ridge_lag4", "trained_snn"):
        ys = [statistics.mean(by_model_fold.get((model, q), [float("nan")])) for q in folds_axis]
        ax.plot(folds_axis, ys, marker="o", label=model)
    ax.set_xlabel("Test fold (chronological quarter of recording)")
    ax.set_ylabel("Velocity R²")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Leave-one-block-out temporal CV (within-recording chronic stand-in)")
    fig.tight_layout()
    fig.savefig(args.fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
