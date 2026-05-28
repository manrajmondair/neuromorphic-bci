#!/bin/bash
#SBATCH --job-name=ncbi-pareto
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/pareto_energy_accuracy-%j.out

# Round 4: energy-accuracy Pareto frontier for trained SNN.
# Sweep {hidden_dim} x {k_history} grid, train trained_snn at each cell,
# count actual input/hidden synops on the test split, and map to per-chip
# energy (CPU x86, A100, Loihi 2, NorthPole).
#
# 5 hidden_dims x 4 k_histories x 3 seeds = 60 trained_snn fits.
# ~60s per fit on H100 -> ~1 hour. Output: a 2D (R^2, synops/pred) cloud
# that becomes the paper's headline neuromorphic figure.
#
# Why this matters: the current Loihi 2 estimate is a single (h=128, k=4)
# point. A Pareto cloud across architectural choices shows the
# accuracy/energy trade-off explicitly, which is THE figure neuromorphic
# papers live and die by.

set -euo pipefail
mkdir -p cluster_logs
cd /home/manraj/neuromorphic-bci

srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  --container-mounts=/home/manraj/neuromorphic-bci:/workspace \
  --container-workdir=/workspace \
  bash -lc '
    set -euo pipefail
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    pip install --quiet --no-deps nlb-tools==0.0.4

    python scripts/run_pareto_energy_accuracy.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out results/cluster/pareto/pareto_energy_accuracy.json \
      --hidden-dims 32 64 128 256 512 \
      --k-histories 0 2 4 8 \
      --event-budget 1.00 \
      --seeds 0 1 2 \
      --tau-ms 10.0 \
      --threshold 0.30 \
      --epochs 80 \
      --patience 15
  '
