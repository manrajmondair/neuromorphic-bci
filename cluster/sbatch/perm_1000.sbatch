#!/bin/bash
#SBATCH --job-name=ncbi-perm-1000
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=cluster_logs/perm_1000-%j.out

# Permutation test with n_perm=1000 across all 4 event budgets.
# Writes results/cluster/snn/permutation_test.json.
#
# Expected wall time on a single H100: ~90–150 minutes.
# Why this matters: replaces p=0.048 from 20 perms with a tight p-value and a
# credible null distribution histogram. Single biggest stats win available.

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

    mkdir -p results/cluster/snn
    python scripts/run_permutation_test.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out results/cluster/snn/permutation_test.json \
      --event-budgets 1.00 0.50 0.25 0.10 \
      --n-perm 1000 \
      --hidden-dim 256 \
      --tau-ms 10.0 \
      --n-restarts 1 \
      --seed 0
  '
