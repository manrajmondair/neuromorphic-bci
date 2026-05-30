#!/bin/bash
#SBATCH --job-name=ncbi-snn-deep
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=cluster_logs/trained_snn_deep-%j.out

# Trained-SNN deep grid: cross the two load-bearing levers (k_history, tau_ms)
# that single-knob sensitivity found but never combined, plus hidden width.
# Stage 1 picks the best combo on val at f=1.0; stage 2 evaluates it across
# all budgets x 3 seeds. Writes results/cluster/trained_snn_deep/.
#
# 8 stage-1 fits (k{16,20} x tau{20,40} x hidden{256,512}, 1 seed) +
# 12 stage-2 fits (best x 4 budgets x 3 seeds) = 20 deep-history fits.

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
      --out-dir results/cluster/trained_snn_deep \
      --k-histories 8 12 16 \
      --taus 10 12 \
      --hidden-dims 256 512 \
      --stage1-seeds 0 \
      --stage2-seeds 0 1 2 \
      --stage2-budgets 1.0 0.5 0.25 0.1 \
      --epochs 150 --patience 25 --n-boot 300
  '
