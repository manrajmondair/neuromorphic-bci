#!/usr/bin/env bash
# Operator helper for driving the AMP PBC cluster from a laptop with kubectl.
# Usage:
#   ./cluster/local_drive.sh push    # stage repo + processed data on the cluster
#   ./cluster/local_drive.sh submit  # submit smoke + 3 real sbatch jobs
#   ./cluster/local_drive.sh status  # squeue + sacct + tail of latest log
#   ./cluster/local_drive.sh pull    # copy results/cluster/ back to laptop
#
# Edit USER and POD if your SUNet differs or your login pod changes.

set -euo pipefail
USER="${USER:-manraj}"
NAMESPACE="slurm"

pod() {
  kubectl get pod -n "$NAMESPACE" -l "stanford/user=$USER" -o jsonpath='{.items[0].metadata.name}'
}

remote() {
  local pod_name
  pod_name=$(pod)
  kubectl exec -n "$NAMESPACE" "$pod_name" -c login -- runuser -u "$USER" -- bash -lc "$1"
}

cp_up() {
  local src=$1 dest=$2
  local pod_name
  pod_name=$(pod)
  kubectl cp "$src" "$NAMESPACE/$pod_name:$dest" -c login
}

cp_down() {
  local src=$1 dest=$2
  local pod_name
  pod_name=$(pod)
  kubectl cp "$NAMESPACE/$pod_name:$src" "$dest" -c login
}

case "${1:-}" in
  push)
    echo "== Cloning / refreshing repo on the cluster =="
    remote "
      mkdir -p ~/neuromorphic-bci/cluster_logs
      cd ~/neuromorphic-bci 2>/dev/null && git pull --ff-only origin main \
        || git clone https://github.com/manrajmondair/neuromorphic-bci.git ~/neuromorphic-bci
    "

    echo "== Staging processed dataset =="
    if [[ ! -f data/processed/processed_mc_rtt.npz ]]; then
      echo "  local data/processed/processed_mc_rtt.npz missing — run scripts/preprocess_mc_rtt.py first"
      exit 1
    fi
    remote "mkdir -p ~/neuromorphic-bci/data/processed"
    cp_up "data/processed/processed_mc_rtt.npz" "/home/$USER/neuromorphic-bci/data/processed/processed_mc_rtt.npz"
    remote "ls -la ~/neuromorphic-bci/data/processed/"
    ;;

  smoke)
    echo "== Submitting smoke test =="
    remote "cd ~/neuromorphic-bci && sbatch cluster/sbatch/smoke.sbatch"
    ;;

  submit)
    echo "== Submitting the three real jobs =="
    remote "cd ~/neuromorphic-bci && \
      sbatch cluster/sbatch/block_cv_grid.sbatch && \
      sbatch cluster/sbatch/perm_1000.sbatch && \
      sbatch cluster/sbatch/hidden_dim_big.sbatch"
    ;;

  status)
    echo "== squeue (mine) =="
    remote "squeue -u $USER" || true
    echo
    echo "== sacct today =="
    remote "sacct -u $USER -S today --format=JobID,JobName,State,Elapsed,ExitCode" || true
    echo
    echo "== latest log lines =="
    remote 'ls -1t ~/neuromorphic-bci/cluster_logs/*.out 2>/dev/null | head -3 | while read f; do echo "--- $f ---"; tail -n 25 "$f"; done' || true
    ;;

  pull)
    echo "== Pulling results/cluster/ down =="
    mkdir -p results/cluster
    # tar on the cluster, untar locally — kubectl cp on a directory is fragile
    remote "cd ~/neuromorphic-bci && tar czf /tmp/cluster_results.tgz results/cluster"
    cp_down "/tmp/cluster_results.tgz" "/tmp/cluster_results.tgz"
    tar xzf /tmp/cluster_results.tgz -C .
    rm -f /tmp/cluster_results.tgz
    remote "rm -f /tmp/cluster_results.tgz"
    echo "== Done. Contents: =="
    find results/cluster -type f | sort
    ;;

  shell)
    pod_name=$(pod)
    echo "Shelling into $pod_name as $USER ..."
    kubectl exec -it -n "$NAMESPACE" "$pod_name" -c login -- runuser -l "$USER"
    ;;

  *)
    cat <<EOF
Usage: $0 {push|smoke|submit|status|pull|shell}

  push    Clone/update repo on the cluster, kubectl cp the processed npz.
  smoke   Submit the 30-min smoke test sbatch.
  submit  Submit block_cv_grid + perm_1000 + hidden_dim_big in parallel.
  status  squeue + sacct + tail of latest job logs.
  pull    Tar the results/cluster tree on the cluster, copy back, untar.
  shell   Drop into an interactive shell inside the login pod.

Environment:
  USER=$USER  (override if your SUNet differs)
EOF
    exit 2
    ;;
esac
