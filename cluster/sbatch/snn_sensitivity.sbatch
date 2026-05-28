#!/bin/bash
#SBATCH --job-name=ncbi-snn-sens
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/snn_sensitivity-%j.out

# Trained-SNN single-knob sensitivity around the canonical (h=256, tau=10,
# thr=0.30, k_hist=4) operating point, three seeds per cell, f=1.0.
# Writes results/cluster/snn_sensitivity/{summary.json, results.csv}.
#
# Expected wall time on a single H100: ~90-150 min (36 trained_snn fits:
# 4 taus + 4 thresholds + 4 k_histories x 3 seeds).
# Why this matters: shows the trained_snn result isn't fragile against
# the chosen hyperparameters — pre-empts reviewer fishing.

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

    python scripts/run_snn_sensitivity.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out-dir results/cluster/snn_sensitivity \
      --event-budget 1.0 \
      --seeds 0 1 2 \
      --taus 5.0 10.0 20.0 40.0 \
      --thresholds 0.15 0.30 0.50 0.80 \
      --k-histories 0 2 4 8 \
      --n-boot 300
  '
