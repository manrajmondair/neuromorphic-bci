#!/bin/bash
#SBATCH --job-name=ncbi-ridge-lag
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=cluster_logs/ridge_lag_sweep-%j.out

# Ridge lag-bins sweep: {0, 1, 2, 4, 8, 16} x {1.0, 0.5, 0.25, 0.1} event
# budgets with bootstrap CIs (n_boot=1000).
# Writes results/cluster/ridge_lag_sweep/{summary.json, results.csv}.
#
# Expected wall time on a single H100: ~10-20 minutes. Ridge is closed-form
# so the GPU is mostly for the bootstrap step. Cheap; high paper value:
# proves "lag_4 is a sensible plateau choice, not a cherry pick."

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

    python scripts/run_ridge_lag_sweep.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out-dir results/cluster/ridge_lag_sweep \
      --lag-bins 0 1 2 4 8 16 \
      --event-budgets 1.00 0.50 0.25 0.10 \
      --seeds 0 \
      --n-boot 1000
  '
