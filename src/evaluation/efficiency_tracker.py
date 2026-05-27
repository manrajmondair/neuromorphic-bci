"""Secondary efficiency analysis for the ridge baseline.

The proposal commits to reporting implementation-relevant quantities
alongside decoding R²: how many spike events the decoder actually
processes at each event budget and how many dense MAC operations an
event-driven architecture would avoid relative to the dense ridge
baseline. None of this is a primary decoding metric — it justifies the
neuromorphic motivation in the final write-up.

What this module computes per event budget:

  1. Distribution of retained spike events per bin (total, mean, median,
     max, std).

  2. Dense MACs the ridge baseline performs per prediction.
     ŷ = X @ W + b with X ∈ R^(1 × N) and W ∈ R^(N × 2), so each
     prediction costs 2 · N MACs. Across the whole recording that's
     2 · N · num_bins.

  3. Event-driven MAC cost. An event-driven implementation of the same
     linear readout adds W[neuron, :] (2 MACs) into a running accumulator
     on every retained spike, then emits one prediction per bin (2 bias
     adds). Total: 2 · events_total + 2 · num_bins.

  4. MACs avoided = dense − event-driven, plus the fraction avoided.

  5. Per-chip energy estimates. Multiplies the MAC counts above by
     published or vendor-cited per-MAC energy figures (see
     `MAC_ENERGY_PJ`) to convert MACs into picojoules of arithmetic
     energy. We report energy per single prediction (per bin), per
     second of recording, and as a fraction avoided relative to the
     dense CPU baseline. These are arithmetic-energy estimates only —
     they do not include memory traffic, leakage, or wireless TX power,
     so they are the lower bound the paper compares against.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.features.event_budget import apply_event_budget

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Per-MAC energy constants (picojoules per multiply-accumulate)
# -------------------------------------------------------------------------
#
# Sources (rounded order-of-magnitude figures for paper context):
#
#   * `cpu_x86`  — a modern desktop x86 core in float32 ≈ 1 nJ/MAC = 1000 pJ.
#                  Conservative ballpark from typical "GFLOPS per watt"
#                  measurements (~10 GFLOPS/W for a single fully-utilised
#                  core, i.e. 1 nJ per float MAC).
#   * `gpu_a100` — NVIDIA A100 at TF32 / FP32 ≈ 30 pJ/MAC.
#   * `loihi2`   — Intel Loihi 2 (Hala Point family) for a synaptic
#                  spike-driven event ≈ 23 pJ/SynOp, which we treat as
#                  one MAC for direct comparison. (Mayr & Davies 2024
#                  press materials cite ~23 pJ/SynOp at full activity.)
#   * `northpole` — IBM NorthPole reported ~2 pJ/MAC for INT8 inference
#                   (Modha et al. Science 2023).
#
# These are public, vendor-reported, order-of-magnitude numbers. The
# paper should cite them as such. If the user supplies a different
# figure on the CLI the runtime constant wins; the baked-in values are
# for reproducibility.
MAC_ENERGY_PJ: dict[str, float] = {
    "cpu_x86": 1000.0,
    "gpu_a100": 30.0,
    "loihi2": 23.0,
    "northpole": 2.0,
}


# -------------------------------------------------------------------------
# Operation-count primitives
# -------------------------------------------------------------------------


def dense_macs_per_prediction(num_neurons: int, num_outputs: int = 2) -> int:
    """MACs for one dense ridge prediction: ŷ = X @ W + b on X ∈ R^(1 × N)."""
    if num_neurons <= 0 or num_outputs <= 0:
        raise ValueError("num_neurons and num_outputs must be positive")
    return int(num_neurons * num_outputs)


def dense_macs_total(num_neurons: int, num_bins: int, num_outputs: int = 2) -> int:
    """Total dense MACs across the whole recording."""
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    return int(dense_macs_per_prediction(num_neurons, num_outputs) * num_bins)


def event_driven_macs_total(
    events_total: int,
    num_bins: int,
    num_outputs: int = 2,
) -> int:
    """Total MACs for an event-driven linear readout.

    Per retained spike: `num_outputs` MACs to accumulate W[neuron, :] into
    the running output. Per bin emission: `num_outputs` MACs for the bias
    add and accumulator reset.
    """
    if events_total < 0 or num_bins < 0:
        raise ValueError("events_total and num_bins must be non-negative")
    return int(num_outputs * events_total + num_outputs * num_bins)


def macs_avoided(dense: int, event_driven: int) -> dict[str, float]:
    """Dict of (dense, event-driven, avoided, avoided-fraction)."""
    avoided = max(int(dense) - int(event_driven), 0)
    fraction = float(avoided) / float(dense) if dense > 0 else 0.0
    return {
        "dense_macs": int(dense),
        "event_driven_macs": int(event_driven),
        "macs_avoided": int(avoided),
        "macs_avoided_fraction": fraction,
    }


# -------------------------------------------------------------------------
# Energy conversions
# -------------------------------------------------------------------------


def macs_to_energy_pj(macs: int | float, chip: str) -> float:
    """Convert MAC count to arithmetic energy in picojoules for one chip."""
    if chip not in MAC_ENERGY_PJ:
        raise KeyError(f"unknown chip {chip!r}; known: {sorted(MAC_ENERGY_PJ)}")
    return float(macs) * MAC_ENERGY_PJ[chip]


def energy_table(
    dense_macs: int,
    event_driven_macs: int,
    num_bins: int,
    bin_size_ms: int,
) -> dict[str, Any]:
    """Per-chip energy table for one (dense, event-driven) MAC pair.

    All energies are arithmetic only (no memory / leakage / TX).
    Returns:
        per_chip[chip] = {
            dense_energy_uj, event_driven_energy_uj,
            dense_uj_per_pred, event_driven_uj_per_pred,
            dense_uw_avg, event_driven_uw_avg,
            energy_avoided_fraction,
        }
    """
    if num_bins <= 0 or bin_size_ms <= 0:
        raise ValueError("num_bins and bin_size_ms must be positive")
    duration_s = num_bins * bin_size_ms / 1000.0
    out: dict[str, Any] = {"per_chip": {}, "duration_s": float(duration_s)}
    for chip, e_pj in MAC_ENERGY_PJ.items():
        dense_pj = dense_macs * e_pj
        edriven_pj = event_driven_macs * e_pj
        out["per_chip"][chip] = {
            "pj_per_mac": float(e_pj),
            "dense_energy_uj": float(dense_pj) / 1e6,
            "event_driven_energy_uj": float(edriven_pj) / 1e6,
            "dense_uj_per_prediction": float(dense_pj) / 1e6 / num_bins,
            "event_driven_uj_per_prediction": float(edriven_pj) / 1e6 / num_bins,
            "dense_avg_uw": float(dense_pj) / 1e6 / duration_s,
            "event_driven_avg_uw": float(edriven_pj) / 1e6 / duration_s,
            "energy_avoided_fraction": (
                (dense_pj - edriven_pj) / dense_pj if dense_pj > 0 else 0.0
            ),
        }
    return out


# -------------------------------------------------------------------------
# Spike-event accounting
# -------------------------------------------------------------------------


def events_per_bin_stats(event_times: list[np.ndarray]) -> dict[str, Any]:
    """Summary statistics for the per-bin event-count distribution."""
    if not event_times:
        return {
            "events_total": 0,
            "num_bins": 0,
            "events_per_bin_mean": 0.0,
            "events_per_bin_median": 0.0,
            "events_per_bin_std": 0.0,
            "events_per_bin_min": 0,
            "events_per_bin_max": 0,
            "empty_bin_fraction": 0.0,
        }
    counts = np.array([int(t.size) for t in event_times], dtype=np.int64)
    return {
        "events_total": int(counts.sum()),
        "num_bins": int(counts.size),
        "events_per_bin_mean": float(counts.mean()),
        "events_per_bin_median": float(np.median(counts)),
        "events_per_bin_std": float(counts.std(ddof=0)),
        "events_per_bin_min": int(counts.min()),
        "events_per_bin_max": int(counts.max()),
        "empty_bin_fraction": float((counts == 0).mean()),
    }


# -------------------------------------------------------------------------
# Top-level analyses
# -------------------------------------------------------------------------


def profile_event_budget(
    event_times: list[np.ndarray],
    event_neurons: list[np.ndarray],
    fraction: float,
    num_neurons: int,
    bin_size_ms: int,
    num_outputs: int = 2,
) -> dict[str, Any]:
    """Run the event-count, MAC-avoidance, and energy analysis for one budget."""
    if fraction < 1.0:
        et, _ = apply_event_budget(event_times, event_neurons, fraction)
    else:
        et = event_times

    spike_stats = events_per_bin_stats(et)
    num_bins = spike_stats["num_bins"]
    dense_t = dense_macs_total(num_neurons, num_bins, num_outputs)
    event_driven_t = event_driven_macs_total(spike_stats["events_total"], num_bins, num_outputs)
    op_stats = macs_avoided(dense_t, event_driven_t)
    energy = energy_table(dense_t, event_driven_t, num_bins, bin_size_ms)

    return {
        "event_budget": float(fraction),
        **spike_stats,
        **op_stats,
        "energy": energy,
    }


def compute_efficiency_summary(
    processed: dict[str, Any],
    fractions: Sequence[float] = (1.00, 0.50, 0.25, 0.10),
    num_outputs: int = 2,
    model: str = "ridge",
) -> dict[str, Any]:
    """Build the full computational-efficiency summary blob."""
    event_times = processed["event_times"]
    event_neurons = processed["event_neurons"]
    num_neurons = int(processed["num_neurons"])
    bin_size_ms = int(processed.get("bin_size_ms", 50))
    num_bins = len(event_times)

    macs_per_pred = dense_macs_per_prediction(num_neurons, num_outputs)
    dense_t = dense_macs_total(num_neurons, num_bins, num_outputs)

    logger.info(
        "efficiency analysis: %d bins, %d neurons, %d outputs, dense_macs/pred=%d, dense_macs_total=%d",
        num_bins, num_neurons, num_outputs, macs_per_pred, dense_t,
    )

    budget_rows: list[dict[str, Any]] = []
    for f in fractions:
        row = profile_event_budget(
            event_times, event_neurons, f, num_neurons, bin_size_ms, num_outputs
        )
        budget_rows.append(row)
        loihi_uj = row["energy"]["per_chip"]["loihi2"]["event_driven_uj_per_prediction"]
        cpu_uj = row["energy"]["per_chip"]["cpu_x86"]["dense_uj_per_prediction"]
        logger.info(
            "budget f=%.2f: events=%d (%.2f/bin) | dense=%d edriven=%d avoided=%d (%.1f%%) | loihi2=%.3f µJ/pred  cpu=%.3f µJ/pred",
            f, row["events_total"], row["events_per_bin_mean"],
            row["dense_macs"], row["event_driven_macs"], row["macs_avoided"],
            100.0 * row["macs_avoided_fraction"], loihi_uj, cpu_uj,
        )

    return {
        "model": model,
        "dataset": str(processed.get("dataset_name", "NLB_MC_RTT")),
        "bin_size_ms": bin_size_ms,
        "num_neurons": num_neurons,
        "num_outputs": int(num_outputs),
        "num_bins": int(num_bins),
        "macs_per_prediction_dense": int(macs_per_pred),
        "dense_macs_total": int(dense_t),
        "mac_energy_pj": dict(MAC_ENERGY_PJ),
        "budgets": budget_rows,
        "notes": (
            "Dense MAC count = 2·N·T (X @ W + b for X[1,N], W[N,2], T bins). "
            "Event-driven MAC count = 2·E + 2·T (per-spike accumulator update "
            "plus per-bin bias). Energy is arithmetic only — no memory traffic, "
            "leakage, or wireless TX. Per-chip pJ/MAC values are vendor or "
            "literature figures, see mac_energy_pj field."
        ),
    }


def save_efficiency_json(summary: dict[str, Any], path) -> None:
    """Persist the efficiency summary to JSON."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    logger.info("wrote %s", path)
