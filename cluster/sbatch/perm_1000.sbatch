#!/bin/bash
#SBATCH --job-name=ncbi-perm-1000
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=cluster_logs/perm_1000-%j.out

# Permutation test with n_perm=1000 across all 4 event budgets.
# Writes results/cluster/snn/permutation_test.json.
#
# This is the RESERVOIR SNN (numpy-only) — no torch/GPU work is done, so we
# do NOT request --gres=gpu:1. Asking for a GPU here would burn ~3 H100-hours
# of QoS budget on what is a pure CPU workload.
#
# Expected wall time: ~3-4 hours on the CPU side of a slinky node.

set -euo pipefail
mkdir -p cluster_logs
cd /home/manraj/neuromorphic-bci

srun --cpus-per-task=16 \
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
