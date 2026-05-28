#!/bin/bash
#SBATCH --job-name=ncbi-trained-null
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=cluster_logs/trained_snn_null_battery-%j.out

# Round 3: multi-shuffle null battery on the TRAINED SNN (not the reservoir SNN).
# 4 event budgets x 4 shuffle types x 20 perms = 320 BPTT fits at h=256, k_hist=4.
#
# Writes results/cluster/snn_trained/null_battery.json.
#
# Why this matters: the existing results/snn/null_battery.json uses the reservoir
# SNN, but the paper's headline decoder is the trained SNN. Repeating the null
# battery on trained_snn gives the strongest possible "the trained-SNN signal is
# in within-bin order + bin alignment, not identity or precise timing" claim.

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

    python scripts/run_trained_snn_null_battery.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out results/cluster/snn_trained/null_battery.json \
      --event-budgets 1.00 0.50 0.25 0.10 \
      --n-perm 20 \
      --hidden-dim 256 \
      --tau-ms 10.0 \
      --threshold 0.30 \
      --k-history 4 \
      --epochs 80 \
      --patience 15
  '
