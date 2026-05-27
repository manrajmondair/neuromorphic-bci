"""Failure-mode analysis from per-bin prediction npz files.

For each model with a predictions npz at the canonical (f=0.25, seed=0)
configuration, compute:
  * per-bin squared error
  * R^2 conditional on cursor-speed quartile
  * R^2 conditional on |velocity| above / below the median

Writes results/failure_analysis/{model}_failure.json and
results/figures/failure_modes.png.
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

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
logger = logging.getLogger("run_failure_analysis")

QUALITATIVE_BUDGET = 0.25
QUALITATIVE_SEED = 0
DEFAULT_NPZS: dict[str, Path] = {
    "ridge": Path(f"results/ridge/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"),
    "ridge_lag4": Path(f"results/ridge/predictions_f1.00_seed0.npz"),  # only have lag4 at f=1.0
    "latency": Path(f"results/latency/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"),
    "snn": Path(f"results/snn/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"),
    "trained_snn": Path(f"results/trained_snn/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"),
    "snn_shuffle": Path(f"results/controls/predictions_f{QUALITATIVE_BUDGET:.2f}_seed{QUALITATIVE_SEED}.npz"),
}


def _r2_joint(y_true, y_pred):
    mean_v = y_true.mean(axis=0, keepdims=True)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - mean_v) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _r2_global_baseline(y_true_subset, y_pred_subset, global_mean, global_ss_tot):
    """R^2 of a subset measured against the GLOBAL mean and total variance.

    1 - sum_subset (y - y_hat)^2 / overall_ss_tot. This stays interpretable
    even when subsetting to bins with low local variance, because the
    denominator does not collapse with the subset.
    """
    ss_res_subset = ((y_true_subset - y_pred_subset) ** 2).sum()
    return float(1.0 - ss_res_subset / max(global_ss_tot, 1e-12))


def _conditional_r2(y_true, y_pred, mask, *, global_mean, global_ss_tot):
    if mask.sum() < 2:
        return float("nan")
    return _r2_global_baseline(y_true[mask], y_pred[mask], global_mean, global_ss_tot)


def _conditional_rmse_relative(y_true, y_pred, mask, global_std):
    if mask.sum() < 2:
        return float("nan")
    rmse = float(np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean()))
    return rmse / max(global_std, 1e-12)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("results/failure_analysis"))
    p.add_argument("--fig-path", type=Path, default=Path("results/figures/failure_modes.png"))
    p.add_argument("--n-bins", type=int, default=4, help="speed bins for conditional R^2")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT, stream=sys.stdout)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    summary_by_model: dict[str, dict] = {}
    for model, path in DEFAULT_NPZS.items():
        if not path.is_file():
            logger.warning("skipping %s: missing %s", model, path)
            continue
        z = np.load(path, allow_pickle=True)
        y_true = np.asarray(z["y_true"])
        y_pred = np.asarray(z["y_pred"])
        speed = np.linalg.norm(y_true, axis=1)

        global_mean = y_true.mean(axis=0, keepdims=True)
        global_ss_tot = float(((y_true - global_mean) ** 2).sum())
        global_std = float(np.sqrt(((y_true - global_mean) ** 2).mean()))

        # Quartile bins of cursor speed.
        edges = np.quantile(speed, np.linspace(0, 1, args.n_bins + 1))
        bin_idx = np.clip(np.digitize(speed, edges[1:-1]), 0, args.n_bins - 1)

        per_quartile: list[dict] = []
        for q in range(args.n_bins):
            mask = bin_idx == q
            r2 = _conditional_r2(y_true, y_pred, mask,
                                 global_mean=global_mean, global_ss_tot=global_ss_tot)
            rrmse = _conditional_rmse_relative(y_true, y_pred, mask, global_std)
            per_quartile.append({
                "quartile": int(q),
                "speed_lo": float(edges[q]),
                "speed_hi": float(edges[q + 1]),
                "n_bins": int(mask.sum()),
                "r2_global_baseline": r2,
                "rmse_relative_to_global_std": rrmse,
            })

        median_speed = float(np.median(speed))
        out = {
            "model": model,
            "n_test_bins": int(y_true.shape[0]),
            "overall_r2_joint": float(_r2_joint(y_true, y_pred)),
            "median_speed": median_speed,
            "r2_above_median": _conditional_r2(
                y_true, y_pred, speed >= median_speed,
                global_mean=global_mean, global_ss_tot=global_ss_tot,
            ),
            "r2_below_median": _conditional_r2(
                y_true, y_pred, speed < median_speed,
                global_mean=global_mean, global_ss_tot=global_ss_tot,
            ),
            "per_quartile": per_quartile,
        }
        (args.out_dir / f"{model}_failure.json").write_text(json.dumps(out, indent=2))
        summary_by_model[model] = out
        logger.info(
            "%-13s overall=%+.4f  by quartile (global baseline)=%s",
            model, out["overall_r2_joint"],
            [round(r["r2_global_baseline"], 4) for r in per_quartile],
        )
        rows.append(out)

    # Plot: per-quartile R^2 for each model.
    if summary_by_model:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for model, out in summary_by_model.items():
            xs = [(r["speed_lo"] + r["speed_hi"]) / 2 for r in out["per_quartile"]]
            ys = [r["r2_global_baseline"] for r in out["per_quartile"]]
            axes[0].plot(xs, ys, marker="o", label=model)
            rs = [r["rmse_relative_to_global_std"] for r in out["per_quartile"]]
            axes[1].plot(xs, rs, marker="o", label=model)
        axes[0].set_xlabel("Cursor speed (quartile centre)")
        axes[0].set_ylabel("R^2 vs. global-mean baseline")
        axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.4)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=9)
        axes[0].set_title("Variance explained in each speed quartile")

        axes[1].set_xlabel("Cursor speed (quartile centre)")
        axes[1].set_ylabel("RMSE / global std")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best", fontsize=9)
        axes[1].set_title("Relative RMSE in each speed quartile")
        fig.suptitle("Failure-mode analysis: decoder performance vs cursor speed", y=1.02)
        fig.tight_layout()
        fig.savefig(args.fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", args.fig_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
