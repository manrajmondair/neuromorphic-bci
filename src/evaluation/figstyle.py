"""Publication figure style — one consistent visual language for every figure.

Importing this module and calling `apply_style()` sets Matplotlib rcParams for
a clean, journal-grade look (no top/right spines, restrained grid, embedded
fonts in PDF). `DECODERS` is the single source of truth for each decoder's
colour, marker, and label, so a reader learns the encoding once and it holds
across every panel in the repo. Colours are the Okabe–Ito colourblind-safe
palette.

Use:
    from src.evaluation.figstyle import apply_style, style_for, save_fig, panel
    apply_style()
    ax.plot(..., **style_for("trained_snn"))
    save_fig(fig, "results/figures/my_figure")   # writes .png (300 dpi) + .pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe–Ito colourblind-safe palette.
_BLACK = "#000000"
_ORANGE = "#E69F00"
_SKY = "#56B4E9"
_GREEN = "#009E73"
_BLUE = "#0072B2"
_VERMILLION = "#D55E00"
_PURPLE = "#CC79A7"
_GREY = "#999999"

# Canonical per-decoder style. `key` is the model id; aliases map several
# script-level names onto one canonical entry so colours never drift.
DECODERS: dict[str, dict] = {
    "ridge":        {"label": "Ridge (counts)",        "color": _GREY,       "marker": "o", "z": 2},
    "latency":      {"label": "First-spike latency",   "color": _ORANGE,     "marker": "v", "z": 2},
    "ridge_hist":   {"label": "Ridge + history",       "color": _BLUE,       "marker": "^", "z": 3},
    "reservoir_snn":{"label": "Reservoir SNN",         "color": _GREEN,      "marker": "s", "z": 4},
    "deeper_snn":   {"label": "Deeper SNN",            "color": _PURPLE,     "marker": "D", "z": 4},
    "trained_snn":  {"label": "Trained SNN",           "color": _VERMILLION, "marker": "P", "z": 5},
    "shuffle":      {"label": "Within-bin shuffle",    "color": "#BBBBBB",   "marker": "x", "z": 1},
}

# Map the many script-level model ids onto the canonical keys above.
_ALIASES = {
    "ridge": "ridge", "ridge_counts": "ridge",
    "latency": "latency",
    "ridge_lag4": "ridge_hist", "ridge_lag20": "ridge_hist",
    "ridge_lag24": "ridge_hist", "ridge_hist": "ridge_hist",
    "snn": "reservoir_snn", "reservoir_snn": "reservoir_snn",
    "deeper_snn": "deeper_snn", "deeper_snn_deep": "deeper_snn",
    "trained_snn": "trained_snn", "trained_snn_deep": "trained_snn",
    "snn_shuffle": "shuffle", "shuffle": "shuffle",
}


def canonical(model: str) -> str:
    return _ALIASES.get(model, model)


def style_for(model: str, *, line: bool = True) -> dict:
    """kwargs for ax.plot/errorbar for a decoder (colour, marker, label, zorder)."""
    d = DECODERS[canonical(model)]
    kw = {"color": d["color"], "marker": d["marker"], "label": d["label"], "zorder": d["z"]}
    if d["label"] == "Within-bin shuffle":
        kw["linestyle"] = "--"
    return kw


def label_for(model: str) -> str:
    return DECODERS[canonical(model)]["label"]


def color_for(model: str) -> str:
    return DECODERS[canonical(model)]["color"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed editable TrueType in PDF (journal req.)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.9,
        "lines.markersize": 5.5,
        "lines.markeredgewidth": 0.0,
    })


def panel(ax, letter: str, *, dx: float = -0.02, dy: float = 0.04) -> None:
    """Bold panel label (a, b, ...) just outside the top-left of the axes."""
    ax.text(dx, 1.0 + dy, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="right")


def save_fig(fig, stem: str | Path) -> None:
    """Save a figure as both <stem>.png (300 dpi) and <stem>.pdf (vector)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
