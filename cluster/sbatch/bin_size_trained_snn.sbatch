#!/bin/bash
#SBATCH --job-name=ncbi-bin-trained
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/bin_size_trained_snn-%j.out

# Round 4: bin-size sweep for the TRAINED SNN.
# Existing results/bin_sweep/* covers ridge + reservoir SNN at 20/50/100 ms;
# this one fills the trained-SNN slice. The order/latency-gap claim and
# the bin-width choice (50 ms) should both hold for the headline decoder.
#
# 3 bin sizes x 4 event budgets x 3 seeds = 36 trained_snn fits. ~60-90 min
# on H100 plus per-bin preprocessing (one-time, ~10 sec each).
#
# Writes results/cluster/bin_sweep_trained/b{20,50,100}/* and summary.json.

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

    python scripts/run_bin_size_trained_snn.py \
      --bin-sizes-ms 20 50 100 \
      --raw-dir data/raw \
      --processed-root data/processed \
      --out-root results/cluster/bin_sweep_trained \
      --event-budgets 1.00 0.50 0.25 0.10 \
      --seeds 0 1 2 \
      --hidden-dim 256 \
      --k-history 4 \
      --threshold 0.30 \
      --epochs 80 \
      --patience 15 \
      --n-boot 300
  '
