#!/bin/bash
#SBATCH --job-name=ncbi-block-cv
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/block_cv-%j.out

# Block CV grid: 10 seeds × 4 event budgets × 4 folds × {ridge, ridge_lag4, trained_snn}.
# Writes results/cluster/block_cv/block_cv.json (streams partial results after every cell).
#
# Expected wall time on a single H100: ~75–90 minutes.
# Why this matters: turns the headline block_cv result from a 1-seed scaffold into
# a 10-seed mean ± std panel sweeping every event budget.

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

    python scripts/run_block_cv.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out-dir results/cluster/block_cv \
      --fig-path results/cluster/figures/block_cv_grid.png \
      --n-folds 4 \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --event-budgets 1.00 0.50 0.25 0.10 \
      --hidden-dim 256 \
      --k-history 4 \
      --threshold 0.30 \
      --epochs 80 \
      --patience 15 \
      --n-boot 1000
  '
