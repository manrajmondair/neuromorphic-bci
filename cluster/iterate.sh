#!/usr/bin/env bash
# Round wrap-up helper: pull cluster results -> regenerate summary -> commit + push.
# Run after a batch of SLURM jobs lands. The aggregator is idempotent so re-running
# is safe; the commit is skipped if nothing changed.
#
# Usage:
#   ./cluster/iterate.sh "round N: short message"
#
# Picks up whichever of these are present under results/cluster/:
#   block_cv/, snn/, hidden_dim_sweep/, trained_snn_ensemble/,
#   ridge_lag_sweep/, snn_sensitivity/, figures/, summary.md.

set -euo pipefail
cd "$(dirname "$0")/.."

msg="${1:-cluster: pull + regenerate summary}"

echo "== pulling results/cluster from cluster =="
.venv/bin/python cluster/cluster_drive.py pull

echo "== running aggregator =="
.venv/bin/python scripts/aggregate_cluster_results.py

echo "== current cluster tree =="
find results/cluster -type f | sort

if [[ -n "$(git status --porcelain results/cluster cluster_logs 2>/dev/null)" ]]; then
  git add results/cluster cluster_logs 2>/dev/null || true
  git status --short results/cluster cluster_logs
  git commit -m "$msg"
  git push origin main
else
  echo "no new cluster artifacts to commit"
fi
