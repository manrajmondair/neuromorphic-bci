"""Generate the full publication figure set from current data, one consistent
style (see src/evaluation/figstyle.py). Any figure whose inputs are missing is
skipped with a warning, so this runs cleanly mid-pipeline.

Figures (results/figures/):
  fig1_frontier            accuracy vs event budget — matched window (a) + best config (b)
  fig2_context_depth       R² vs history depth — every decoder converges (context is the lever)
  fig3_snn_context         trained SNN: history on the input saturates, on the readout it scales
  fig4_energy_accuracy     accuracy vs energy/prediction — the neuromorphic payoff
  fig5_reconstruction      predicted vs true velocity + integrated cursor path (best decoder)
  fig6_cv_robustness       per-fold blocked-CV R² — stable across the recording
"""
from __future__ import annotations

import csv
import json
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.figstyle import (apply_style, canonical, color_for, label_for,
                                     panel, save_fig, style_for, DECODERS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(message)s")
logger = logging.getLogger("paper_figures")
REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "results" / "figures"
BUDGETS = [1.0, 0.5, 0.25, 0.1]


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _read_csv(p: Path):
    return list(csv.DictReader(p.open())) if p.exists() else None


def _agg(rows, keyf, valf="r2_joint"):
    """group rows -> {key: (mean, std)} over the value field."""
    g = defaultdict(list)
    for r in rows:
        g[keyf(r)].append(float(r[valf]))
    return {k: (st.mean(v), st.pstdev(v)) for k, v in g.items()}


def _budget_axis(ax):
    ax.set_xlabel("Event budget  $f$  (fraction of earliest spikes kept)")
    ax.set_xlim(1.05, 0.03)
    ax.set_xticks([1.0, 0.75, 0.5, 0.25, 0.1])


# --------------------------------------------------------------------------- #
def fig1_frontier():
    bcv = _read_json(REPO / "results/cluster/block_cv/block_cv.json")
    best = _read_json(REPO / "results/best/best_models.json")
    if not bcv or not best:
        logger.warning("fig1: missing block_cv or best_models"); return
    rows = bcv["rows"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)

    # (a) matched-window blocked CV
    order_a = ["ridge", "ridge_lag4", "deeper_snn", "reservoir_snn", "trained_snn"]
    for m in order_a:
        sub = [r for r in rows if r["model"] == m]
        if not sub:
            continue
        stats = _agg(sub, lambda r: round(float(r["event_budget"]), 2))
        xs = [b for b in BUDGETS if b in stats]
        ys = [stats[b][0] for b in xs]; es = [stats[b][1] for b in xs]
        kw = style_for(m)
        axL.fill_between(xs, np.array(ys) - np.array(es), np.array(ys) + np.array(es),
                         color=kw["color"], alpha=0.12, zorder=kw["zorder"] - 1)
        axL.plot(xs, ys, **kw)
    _budget_axis(axL); axL.set_ylabel("Velocity decode  $R^2$")
    axL.set_title("Matched 200 ms window")
    axL.set_ylim(-0.03, 0.72)
    panel(axL, "a")
    axL.annotate("history-using\ndecoders coincide", xy=(0.84, 0.52), xytext=(0.62, 0.30),
                 fontsize=8.5, color="#444", ha="center",
                 arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))

    # (b) best configuration
    order_b = ["ridge", "latency", "deeper_snn", "ridge_hist", "reservoir_snn", "trained_snn"]
    for label_key in order_b:
        # best_models keys are display labels; map back
        match = next((lbl for lbl in best if canonical_from_label(lbl) == label_key), None)
        if match is None:
            continue
        cells = best[match]
        xs = [b for b in BUDGETS if f"f={b}" in cells]
        ys = [cells[f"f={b}"]["mean"] for b in xs]
        kw = style_for(label_key)
        axR.plot(xs, ys, **kw)
    _budget_axis(axR); axR.set_title("Best per-decoder context")
    panel(axR, "b")
    axR.legend(loc="upper right", ncol=1, fontsize=8)
    fig.tight_layout()
    save_fig(fig, FIGDIR / "fig1_frontier")
    logger.info("wrote fig1_frontier")


def canonical_from_label(lbl: str) -> str:
    """Map a best_models display label back to a canonical decoder key."""
    l = lbl.lower()
    if "trained" in l: return "trained_snn"
    if "deeper" in l: return "deeper_snn"
    if "reservoir" in l: return "reservoir_snn"
    if "ridge + deep" in l or "history" in l: return "ridge_hist"
    if "latency" in l: return "latency"
    if "ridge" in l: return "ridge"
    return l


# --------------------------------------------------------------------------- #
def fig2_context_depth():
    rows = _read_csv(REPO / "results/sweeps/history_depth.csv")
    if not rows:
        logger.warning("fig2: missing history_depth.csv"); return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    # ridge family spans k=0 (memoryless counts) -> k>0 (ridge+history); one curve.
    families = [("ridge_hist", {"ridge", "ridge_hist"}),
                ("reservoir_snn", {"reservoir_snn"}),
                ("trained_snn", {"trained_snn"})]
    for m, members in families:
        sub = [r for r in rows if canonical(r["model"]) in members]
        if not sub:
            continue
        stats = _agg(sub, lambda r: int(r["k_bins"]))
        ks = sorted(stats)
        xs = [k * 50 for k in ks]  # ms
        ys = [stats[k][0] for k in ks]; es = [stats[k][1] for k in ks]
        kw = style_for(m)
        ax.fill_between(xs, np.array(ys) - np.array(es), np.array(ys) + np.array(es),
                        color=kw["color"], alpha=0.12, zorder=kw["zorder"] - 1)
        ax.plot(xs, ys, **kw)
    ax.set_xlabel("History window  (ms)")
    ax.set_ylabel("Velocity decode  $R^2$  ($f = 1.0$)")
    ax.set_ylim(-0.03, 0.72)
    ax.legend(loc="lower right")
    ax.annotate("memoryless", xy=(0, 0.12), xytext=(120, -0.0), fontsize=8.5, color="#444",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    fig.tight_layout()
    save_fig(fig, FIGDIR / "fig2_context_depth")
    logger.info("wrote fig2_context_depth")


# --------------------------------------------------------------------------- #
def fig3_snn_context():
    rows = _read_csv(REPO / "results/sweeps/trained_snn_context.csv")
    if not rows:
        logger.warning("fig3: missing trained_snn_context.csv"); return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    styles = {"readout_lag": dict(color=DECODERS["trained_snn"]["color"], marker="P",
                                  label="history at the readout"),
              "input_history": dict(color="#888888", marker="o",
                                    label="history at the input")}
    for mech, sk in styles.items():
        sub = [r for r in rows if r["mechanism"] == mech]
        if not sub:
            continue
        stats = _agg(sub, lambda r: int(r["k_bins"]))
        ks = sorted(stats); xs = [k * 50 for k in ks]
        ys = [stats[k][0] for k in ks]; es = [stats[k][1] for k in ks]
        ax.fill_between(xs, np.array(ys) - np.array(es), np.array(ys) + np.array(es),
                        color=sk["color"], alpha=0.12)
        ax.plot(xs, ys, color=sk["color"], marker=sk["marker"], label=sk["label"])
    ax.set_xlabel("History window  (ms)")
    ax.set_ylabel("Trained-SNN decode  $R^2$  ($f = 1.0$)")
    ax.set_ylim(-0.03, 0.72)
    ax.legend(loc="upper left", title="trained LIF SNN")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "fig3_snn_context")
    logger.info("wrote fig3_snn_context")


# --------------------------------------------------------------------------- #
def fig4_energy_accuracy():
    pj = _read_json(REPO / "results/cluster/pareto/pareto_energy_accuracy.json")
    if not pj:
        logger.warning("fig4: missing pareto data"); return
    rows = [r for r in pj["rows"] if abs(float(r["event_budget"]) - 1.0) < 1e-6]
    # mean over seeds per (hidden, k_history); collect each hardware's energy.
    hw = ["cpu_x86", "gpu_a100", "loihi2", "northpole"]
    g = defaultdict(lambda: defaultdict(list))
    r2g = defaultdict(list)
    for r in rows:
        key = (int(r["hidden_dim"]), int(r["k_history"]))
        r2g[key].append(float(r["r2_joint"]))
        for h in hw:
            g[key][h].append(float(r["energy_pj_per_prediction"][h]))
    keys = list(r2g)
    r2 = {k: np.mean(r2g[k]) for k in keys}

    hw_style = {
        "cpu_x86":  ("CPU (x86)",        "#444444", "o"),
        "gpu_a100": ("GPU (A100)",       "#0072B2", "^"),
        "loihi2":   ("Loihi-2",          "#D55E00", "s"),
        "northpole":("NorthPole",        "#009E73", "D"),
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    def pareto(xy):
        xy = sorted(xy)
        out = []; best = -np.inf
        for x, y in xy:
            if y > best:
                best = y; out.append((x, y))
        return out

    for h in hw:
        e_nj = {k: np.mean(g[k][h]) / 1e3 for k in keys}
        label, color, marker = hw_style[h]
        ax.scatter([e_nj[k] for k in keys], [r2[k] for k in keys], s=24, color=color,
                   alpha=0.45, marker=marker, zorder=2, edgecolor="none")
        fr = pareto([(e_nj[k], r2[k]) for k in keys])
        ax.plot([p[0] for p in fr], [p[1] for p in fr], color=color, marker=marker,
                markersize=5, lw=1.7, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Energy per prediction  (nJ, arithmetic only)")
    ax.set_ylabel("Velocity decode  $R^2$  ($f = 1.0$)")
    ax.set_ylim(-0.03, 0.62)
    ax.grid(True, which="both", axis="x")
    ax.legend(loc="center right", title="hardware target")
    ax.text(0.03, 0.10, "Same trained-SNN models on each target — neuromorphic\n"
            "hardware shifts the accuracy frontier ~10–500× lower in energy.",
            transform=ax.transAxes, fontsize=8.2, color="#555", va="bottom", ha="left")
    fig.tight_layout()
    save_fig(fig, FIGDIR / "fig4_energy_accuracy")
    logger.info("wrote fig4_energy_accuracy")


# --------------------------------------------------------------------------- #
def fig5_reconstruction():
    # Prefer the best decoder (trained SNN) at full budget; fall back to reservoir.
    npz = REPO / "results/trained_snn/predictions_best_f1.00.npz"
    model = "trained_snn"
    if not npz.exists():
        npz = REPO / "results/snn/predictions_f0.25_seed0.npz"; model = "reservoir_snn"
    if not npz.exists():
        logger.warning("fig5: missing predictions"); return
    d = np.load(npz, allow_pickle=True)
    yt, yp = np.asarray(d["y_true"]), np.asarray(d["y_pred"])
    bin_ms = int(d["bin_size_ms"]); f = float(d["event_budget"])
    col = color_for(model)
    fig = plt.figure(figsize=(9.2, 3.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], hspace=0.18, wspace=0.28)

    n = min(200, yt.shape[0]); t = np.arange(n) * bin_ms / 1000.0
    for i, name in enumerate(["$v_x$", "$v_y$"]):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, yt[:n, i], color="black", lw=1.2, label="measured", zorder=2)
        ax.plot(t, yp[:n, i], color=col, lw=1.2, label="decoded", alpha=0.9, zorder=3)
        ax.set_ylabel(f"{name} (mm/s)")
        ax.grid(True, axis="y")
        if i == 0:
            ax.legend(loc="upper right", ncol=2); panel(ax, "a")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time  (s)")

    # integrated 2-D cursor path over the same window
    axp = fig.add_subplot(gs[:, 1])
    dt = bin_ms / 1000.0
    pt = np.cumsum(yt[:n] * dt, axis=0); pp = np.cumsum(yp[:n] * dt, axis=0)
    axp.plot(pt[:, 0], pt[:, 1], color="black", lw=1.3, label="measured", zorder=2)
    axp.plot(pp[:, 0], pp[:, 1], color=col, lw=1.3, label="decoded", alpha=0.9, zorder=3)
    axp.scatter([pt[0, 0]], [pt[0, 1]], color="black", s=18, zorder=4)
    axp.set_xlabel("cursor $x$ (mm)"); axp.set_ylabel("cursor $y$ (mm)")
    axp.set_aspect("equal", adjustable="datalim"); axp.grid(True)
    panel(axp, "b")
    r2 = 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum()
    fig.text(0.01, 0.99, f"{label_for(model)} reconstruction   ($f$ = {f:g},  $R^2$ = {r2:.2f})",
             ha="left", va="top", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, FIGDIR / "fig5_reconstruction")
    logger.info("wrote fig5_reconstruction")


# --------------------------------------------------------------------------- #
def fig6_cv_robustness():
    bcv = _read_json(REPO / "results/cluster/block_cv/block_cv.json")
    if not bcv:
        logger.warning("fig6: missing block_cv"); return
    rows = [r for r in bcv["rows"] if abs(float(r["event_budget"]) - 1.0) < 1e-6]
    folds = sorted({int(r["fold"]) for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for m in ["ridge", "ridge_lag4", "deeper_snn", "reservoir_snn", "trained_snn"]:
        sub = [r for r in rows if r["model"] == m]
        if not sub:
            continue
        stats = _agg(sub, lambda r: int(r["fold"]))
        xs = [fdx for fdx in folds if fdx in stats]
        ys = [stats[fdx][0] for fdx in xs]
        kw = style_for(m); kw.pop("zorder", None)
        ax.plot(xs, ys, **kw)
    ax.set_xlabel("Test fold  (chronological quarter of recording)")
    ax.set_ylabel("Velocity decode  $R^2$  ($f = 1.0$)")
    ax.set_xticks(folds)
    ax.set_ylim(0.0, 0.62)
    ax.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    save_fig(fig, FIGDIR / "fig6_cv_robustness")
    logger.info("wrote fig6_cv_robustness")


def main() -> int:
    apply_style()
    fig1_frontier()
    fig2_context_depth()
    fig3_snn_context()
    fig4_energy_accuracy()
    fig5_reconstruction()
    fig6_cv_robustness()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
