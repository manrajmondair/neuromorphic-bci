#!/bin/bash
#SBATCH --job-name=ncbi-chan-drop
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/channel_dropout-%j.out

# Round 4: test-time channel dropout — simulate post-deployment electrode failure.
# Train ridge_lag4 + trained_snn ONCE on clean data, then evaluate on the test
# split with a fraction p of channels zeroed out, repeated across K random
# masks per fraction.
#
# 5 fractions x 10 masks x 2 models = 100 prediction passes after 2 training
# fits. Trained SNN training dominates wall time (~3-5 min); the prediction
# passes are cheap. Total ~30-60 minutes on H100.
#
# Why this matters: a key implantable-BCI claim is that neuromorphic decoders
# are deployable under electrode failure. Showing trained_snn degrades more
# gracefully than ridge_lag4 under random channel loss is a direct
# implantable-relevance argument.

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

    python scripts/run_channel_dropout.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out results/cluster/channel_dropout/results.json \
      --dropout-fractions 0.0 0.1 0.2 0.3 0.5 \
      --n-masks 10 \
      --event-budget 1.00 \
      --train-seed 0 \
      --hidden-dim 256 \
      --k-history 4 \
      --threshold 0.30 \
      --epochs 80 \
      --patience 15
  '
