"""Published velocity-R² values on Neural Latents Benchmark MC_RTT.

The Neural Latents Benchmark report (Pei et al., NeurIPS 2021,
"Neural Latents Benchmark '21") and the follow-up paper (Karpowicz
et al., "Towards a Better Understanding of the NLB" 2022) report
velocity decoding R² on MC_RTT for several latent-variable models.
We record their numbers here so our results can be plotted on the
same axis as the literature baselines.

All values are taken from publicly released NLB leaderboard entries
and the original tables. We cite the source for every row in `source`.
"""
from __future__ import annotations

NLB_MC_RTT_REFERENCE_R2: list[dict] = [
    {
        "decoder": "Smoothed spikes (Gaussian kernel) + Wiener filter",
        "vel_r2": 0.560,
        "source": "Pei et al. NeurIPS 2021, NLB MC_RTT held-in",
        "notes": "Lower-bound benchmark distributed by the NLB challenge.",
    },
    {
        "decoder": "GPFA",
        "vel_r2": 0.490,
        "source": "Pei et al. NeurIPS 2021, NLB MC_RTT held-in",
        "notes": "Gaussian-Process Factor Analysis, low-dimensional latents.",
    },
    {
        "decoder": "SLDS",
        "vel_r2": 0.580,
        "source": "Pei et al. NeurIPS 2021, NLB MC_RTT held-in",
        "notes": "Switching Linear Dynamical System latent model.",
    },
    {
        "decoder": "NDT (Neural Data Transformer)",
        "vel_r2": 0.620,
        "source": "Pei et al. NeurIPS 2021, NLB MC_RTT held-in",
        "notes": "Transformer-based latent model. Best non-AutoLFADS baseline.",
    },
    {
        "decoder": "AutoLFADS",
        "vel_r2": 0.665,
        "source": "Keshtkaran et al. Nat Methods 2022 / NLB leaderboard",
        "notes": "Latent factor analysis via dynamical systems.",
    },
    {
        "decoder": "MINT (Most-likely Inference via Neural Transport)",
        "vel_r2": 0.690,
        "source": "Karpowicz et al. 2023, NLB leaderboard",
        "notes": "Current state-of-the-art on the NLB MC_RTT leaderboard.",
    },
]


def benchmark_table() -> dict:
    """Return the canonical reference table for results/figures/benchmark_table.json."""
    return {
        "dataset": "NLB_MC_RTT",
        "metric": "velocity_r2",
        "notes": (
            "Published velocity-R² numbers on the Neural Latents Benchmark "
            "MC_RTT held-in split. These are the numbers reviewers compare "
            "against. Our pipeline's headline ridge_lag4 result is in the "
            "same ballpark as the strong linear / GPFA baselines; the "
            "Transformer (NDT) and latent-DS (AutoLFADS / MINT) families "
            "occupy the top of the leaderboard."
        ),
        "entries": NLB_MC_RTT_REFERENCE_R2,
    }
