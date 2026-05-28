#!/bin/bash
#SBATCH --job-name=ncbi-snn-ensemble
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=cluster_logs/snn_ensemble-%j.out

# Trained-SNN 10-seed ensemble across all 4 event budgets.
# Writes results/cluster/trained_snn_ensemble/{individual.csv, ensemble.json,
# ensemble_predictions_f*.npz}.
#
# Expected wall time on a single H100: ~2-3 hours (40 trained_snn fits at
# hidden=256, k_history=4, epochs<=100).
# Why this matters: BPTT has known initialization variance. Averaging 10
# independent fits gives a bagging-style accuracy gain *and* a credible
# variance estimate for the headline trained_snn number.

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

    python scripts/run_trained_snn_ensemble.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out-dir results/cluster/trained_snn_ensemble \
      --event-budgets 1.00 0.50 0.25 0.10 \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --hidden-dim 256 \
      --tau-ms 10.0 \
      --threshold 0.30 \
      --k-history 4 \
      --epochs 100 \
      --patience 18 \
      --n-boot 1000
  '
