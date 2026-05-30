#!/bin/bash
#SBATCH --job-name=ncbi-snn-subbin
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=cluster_logs/trained_snn_subbin-%j.out

# The deep-grid found the trained SNN peaks at k=8 and degrades at k>=12:
# BPTT over the long sub-bin sequence (k=16 @ 10 sub-bins = 170 steps) won't
# train. The limit is sequence LENGTH, not context. Coarsen the within-bin
# resolution (num_sub_bins 4/6) so deep k_history stays a short, trainable
# sequence -- within-bin timing barely carries signal anyway. Cross
# num_sub_bins x k_history to find where deep context becomes trainable.
# Writes results/cluster/trained_snn_subbin/.

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

    python scripts/run_trained_snn_grid.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out-dir results/cluster/trained_snn_subbin \
      --k-histories 8 12 16 20 \
      --taus 12 \
      --hidden-dims 512 \
      --num-sub-bins-list 4 6 10 \
      --stage1-seeds 0 \
      --stage2-seeds 0 1 2 \
      --stage2-budgets 1.0 0.5 0.25 0.1 \
      --epochs 150 --patience 25 --n-boot 300
  '
