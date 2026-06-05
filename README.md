# Sparse Event-Driven Neuromorphic Decoders for Implantable BCIs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Dataset: NLB MC_RTT](https://img.shields.io/badge/dataset-NLB%20MC__RTT-success.svg)](docs/dataset.md)

**Manraj Mondair, Alexander Soto** · Stanford University

A controlled study of how much of a motor-cortex velocity decode survives as the spike stream is made
progressively sparser, and whether an event-driven spiking neural network (SNN) can match a strong
linear decoder while exposing a path to low-energy implantable hardware. Decoders run on the Neural
Latents Benchmark **MC_RTT** dataset under a multi-seed, leakage-safe, blocked cross-validation grid.

## Paper

**[Sparse Event-Driven Neuromorphic Decoders for Implantable BCIs (PDF)](mondair_soto_neuromorphic_bci.pdf)**
— the full write-up: methods, controls, results, and the energy analysis.

<p align="center">
  <img src="results/figures/fig1_frontier.png" width="760"
       alt="Velocity decode R² versus sparse event budget, matched-window and best-config."><br>
  <em>At a matched ≈200&nbsp;ms history window every history-using decoder coincides — temporal context,
  not decoder class, sets the accuracy (a). Given each decoder its own context window, the spiking
  decoders pull ahead, the trained LIF SNN leading at every event budget (b).</em>
</p>

## Headline results

- **Temporal context, not within-bin spike timing, carries the decode.** A memoryless decoder reaches
  R² ≈ 0.17; ≈200 ms of history triples it to 0.51–0.54, where the linear, reservoir-SNN, and
  trained-SNN decoders are statistically indistinguishable. Null controls show the decode rides on
  per-bin firing rate and its temporal alignment to movement — not spike order, precise timing, or
  neuron identity.
- **Given deep context, the SNNs lead on accuracy *and* energy.** A trained LIF SNN with a lag-stacked
  readout reaches **R² = 0.68 at full budget — the best decoder at every event budget** (reservoir SNN
  0.64 just behind), reaching the latent-dynamics tier of the leaderboard while costing tens of
  nanojoules per prediction on Loihi-2-class hardware — 40–500× below a dense CPU readout.

## Repository layout

```
src/data/         NLB MC_RTT loading, binning, velocity, train/val/test split
src/features/     spike-count, lag-history, latency, event-budget, jitter, causal-window
src/models/       ridge, latency, trained LIF SNN (BPTT), reservoir SNN, deeper SNN, readouts
src/controls/     order-shuffle and the multi-null battery
src/evaluation/   R² metric, efficiency/energy model, experiment runner, plots
src/utils/        seeding and a mock-data generator (schema-identical to real)
scripts/          one CLI per experiment (run with --help)
notebooks/        Colab GPU walkthrough of the ridge baseline
docs/             dataset provenance and the shared data-interface contract
cluster/          SLURM job scripts + operator docs for the H100 grids
results/          per-experiment CSV/JSON, prediction npz, and final figures
tests/            invariant and smoke tests (run on mock data, no download needed)
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps nlb-tools                 # upstream pandas pin clash — see docs/dataset.md

python scripts/download_mc_rtt.py               # -> data/raw/000129/*.nwb
python scripts/preprocess_mc_rtt.py             # -> data/processed/processed_mc_rtt.npz

python scripts/run_ridge.py                     # count + history linear baselines
python scripts/run_snn.py                       # reservoir SNN + shuffle control
python scripts/run_trained_snn_grid.py          # trained SNN (best decoder)
python scripts/aggregate_best_models.py         # best-of-every-model table + figure
python scripts/run_efficiency_analysis.py       # MAC / energy accounting
python scripts/generate_paper_figures.py        # publication figure set (fig1–6)

pytest tests/                                   # invariant + smoke tests (no download needed)
```

The multi-seed grids, permutation test, and energy Pareto were produced on a 32× H100 SLURM cluster;
the job scripts and operator playbook are in [`cluster/`](cluster/), and those outputs live under
[`results/cluster/`](results/cluster/). The SNN side can be developed without the download via
`src.utils.mock_data.make_mock_processed_data()`.

## Dataset

Neural Latents Benchmark **MC_RTT** (primary motor cortex, self-paced random target reaching; DANDI
dandiset `000129`). Provenance, citation, and the two-line download are in
[`docs/dataset.md`](docs/dataset.md); the processed-data contract every model depends on is in
[`docs/data_interface.md`](docs/data_interface.md). Raw and processed data are never committed.

## Citation

If you use this code, please cite the repository ([`CITATION.cff`](CITATION.cff)) and the dataset
(Pei et al., *Neural Latents Benchmark*, NeurIPS 2021).

## License

Released under the [MIT License](LICENSE). The MC_RTT dataset retains its own DANDI license and is not
redistributed here.
