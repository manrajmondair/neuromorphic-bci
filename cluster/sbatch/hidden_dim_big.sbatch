#!/bin/bash
#SBATCH --job-name=ncbi-hidden-big
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=cluster_logs/hidden_dim_big-%j.out

# Hidden-dim scaling extended to 1024 and 2048 with 3 seeds.
# Writes results/cluster/hidden_dim_sweep/h{1024,2048}_results.{json,csv}
# and an updated summary.json combining the existing {32,64,128,256,512} runs
# with the new wide ones.
#
# Expected wall time on a single H100: ~30–60 minutes.
# Why this matters: existing saturation claim ("plateau ~256") was reported
# from a T4 budget. Extending to 1024/2048 confirms the plateau holds on
# big-iron and shuts down a likely reviewer question.

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

    python scripts/run_hidden_dim_sweep.py \
      --hidden-dims 1024 2048 \
      --event-budgets 1.00 0.25 \
      --seeds 0 1 2 \
      --k-history 4 \
      --epochs 100 \
      --patience 18 \
      --n-boot 300 \
      --out-dir results/cluster/hidden_dim_sweep
  '
