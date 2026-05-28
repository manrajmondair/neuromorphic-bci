#!/bin/bash
#SBATCH --job-name=ncbi-smoke
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=cluster_logs/smoke-%j.out

# 30-minute smoke test:
#   1. Verify GPU + container import succeed.
#   2. pip install the repo's deps + nlb-tools inside the container.
#   3. Verify torch.cuda + pytest collect + a tiny block_cv run on f=1.0,
#      1 fold, 1 seed, skip SNN. This proves the full path works before
#      we commit hours of GPU budget to the real grid.

set -euo pipefail
mkdir -p cluster_logs
cd /home/manraj/neuromorphic-bci

srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  --container-mounts=/home/manraj/neuromorphic-bci:/workspace \
  --container-workdir=/workspace \
  bash -lc '
    set -euo pipefail
    nvidia-smi
    python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))"

    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    pip install --quiet --no-deps nlb-tools==0.0.4

    python -m pytest tests/ -q --maxfail=1
    mkdir -p results/cluster/smoke
    python scripts/run_block_cv.py \
      --processed-path data/processed/processed_mc_rtt.npz \
      --out-dir results/cluster/smoke \
      --fig-path results/cluster/figures/smoke_block_cv.png \
      --n-folds 4 \
      --seeds 0 \
      --event-budgets 1.00 \
      --hidden-dim 128 \
      --k-history 4 \
      --epochs 30 \
      --patience 8 \
      --n-boot 100
    echo "SMOKE TEST OK"
  '
