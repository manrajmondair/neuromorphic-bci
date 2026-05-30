#!/bin/bash
#SBATCH --job-name=ncbi-deeper-snn
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=cluster_logs/deeper_snn_deep-%j.out

# Deeper SNN at depth + recurrence + deep history: does a multi-layer
# recurrent LIF beat the single-layer trained SNN once all get long
# context? 2-layer [256,128] recurrent LIF, k_history=16, tau=20, across
# all budgets x 3 seeds. Writes results/cluster/deeper_snn_deep/.

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

    python scripts/run_deeper_snn.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --results-csv results/cluster/deeper_snn_deep/results.csv \
      --results-json results/cluster/deeper_snn_deep/deeper_snn_results.json \
      --event-budgets 1.0 0.5 0.25 0.1 \
      --seeds 0 1 2 \
      --hidden-dims 256 128 \
      --k-history 16 \
      --tau-ms 10 \
      --recurrent \
      --epochs 120 --patience 20 --n-boot 300
  '
