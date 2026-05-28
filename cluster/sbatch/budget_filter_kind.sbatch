#!/bin/bash
#SBATCH --job-name=ncbi-budget-kind
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/budget_filter_kind-%j.out

# Round 4: directly test whether the 'earliest k events' filter rule
# matters — compare earliest vs random vs latest at all 4 event budgets,
# 3 seeds each, for both ridge_lag4 and trained_snn.
#
# 4 budgets x 3 kinds x 3 seeds x {ridge_lag4 (cheap), trained_snn (BPTT)} =
# 108 fits. Trained_snn dominates wall time at ~60s/fit -> ~1.5 hours on H100.
#
# Why this matters: the paper claims sparse event-budget decoding is
# useful and implicitly assumes "earliest k" is the canonical filter.
# If 'random' achieves the same R^2 as 'earliest' at the same budget,
# then the signal is in spike order alone (not initiation timing).
# If 'earliest' beats 'random', then movement-onset structure is also
# carrying weight. Either result is publishable.

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

    python scripts/run_budget_filter_kind.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out results/cluster/budget_filter_kind/results.json \
      --event-budgets 1.00 0.50 0.25 0.10 \
      --kinds earliest random latest \
      --seeds 0 1 2 \
      --hidden-dim 256 \
      --k-history 4 \
      --threshold 0.30 \
      --epochs 80 \
      --patience 15 \
      --n-boot 500
  '
