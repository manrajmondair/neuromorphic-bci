# Neuromorphic BCI: Sparse Event-Based Decoding for Implantable BCI

**Course:** EE207 — Neuromorphics: Brains in Silicon
**Authors:** Manraj Mondair, Alexander Soto

How few spikes are enough? We compare a standard spike-count ridge decoder
against a sparse spike-latency SNN decoder on the Neural Latents Benchmark
MC_RTT motor-cortex dataset, sweeping the fraction of spike events processed
per bin and measuring continuous 2D cursor-velocity R².

## Research question

Can a sparse spike-latency SNN decoder predict continuous cursor velocity
from motor-cortex spikes while using fewer neural events than a standard
spike-count decoder?

## Dataset

Neural Latents Benchmark **MC_RTT** (motor cortex, self-paced random target
reaching). Distributed via DANDI (dandiset `000129`). Full provenance,
citation, and download instructions live in **`docs/dataset.md`**.

```bash
python scripts/download_mc_rtt.py     # writes data/raw/000129/...
python scripts/preprocess_mc_rtt.py   # writes data/processed/processed_mc_rtt.npz
```

## Repo layout

```
configs/        per-experiment YAMLs (bin size, event budgets, seeds)
data/           raw/ and processed/ are gitignored; only README tracked
docs/           the shared data-interface contract lives here — read first
src/data/       NLB MC_RTT loading, binning, velocity, train/val/test split
src/features/   spike-count, latency, order-comparison, event-budget filters
src/models/     ridge decoder, LIF SNN decoder, linear readouts
src/controls/   order-shuffle control
src/evaluation/ R² metric, plots, experiment runner
src/utils/      seeds, mock data generator (lets SNN side dev without real data)
scripts/        one CLI per experiment
notebooks/      exploration and per-model walkthroughs
results/        per-model CSV/JSON outputs and final figures
report/         proposal + final report sources
```

## Work split

| Owner | Branch | Subsystems |
|---|---|---|
| Manraj | `data-ridge-baseline` | `src/data`, `src/features/spike_counts.py`, `src/features/event_budget.py`, `src/models/ridge_decoder.py`, `src/evaluation` |
| Alexander | `snn-latency-decoder` | `src/features/latency_order.py`, `src/models/snn_decoder.py`, `src/models/readout.py`, `src/controls/order_shuffle.py` |

Both sides share the contract in `docs/data_interface.md` and the metric
in `src/evaluation/metrics.py`. Nothing else needs synchronous coordination.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps nlb-tools            # upstream pandas pin clash; see docs/dataset.md
python scripts/download_mc_rtt.py          # writes data/raw/000129/*.nwb
python scripts/preprocess_mc_rtt.py        # writes data/processed/processed_mc_rtt.npz
python scripts/run_ridge.py                # writes results/ridge/*.csv + .json + predictions
python scripts/run_efficiency_analysis.py  # writes results/ridge/computational_efficiency.json
python scripts/run_snn.py                  # writes results/snn/*.csv + .json (Alex's branch)
python scripts/generate_final_figures.py   # writes results/figures/*.png at dpi=300
```

The SNN side can develop without `data/processed/processed_mc_rtt.npz` by importing
`src.utils.mock_data.make_mock_processed_data()` — same schema as the real
output, so swapping is a one-line change.

## Headline deliverable

A single figure: velocity R² versus event budget f ∈ {1.00, 0.50, 0.25, 0.10},
with three lines: ridge, SNN, shuffled-SNN control. Comparison table in
`results/figures/comparison.csv`.
