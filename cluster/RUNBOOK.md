# Cluster runbook

The end-to-end playbook, from a verified kubeconfig to a converged
iteration loop. Run the one-time access setup in `SETUP.md` first, and
see `CLUSTER.md` for the environment and guardrails.

## Phase 0 — verify connectivity (10 s)

```bash
kubectl get pod -n slurm -l stanford/user=$USER
```

Expect one pod whose name starts with `slurm-login-`. If the result is
empty, the LoginSet has not been provisioned yet — contact the cluster
administrator.

## Phase 1 — stage data + repo on the cluster (1–2 min)

```bash
./cluster/local_drive.sh push
```

Clones the repo into `/home/<user>/neuromorphic-bci` on the cluster (or
`git pull --ff-only` if it already exists), then `kubectl cp`s the local
`data/processed/processed_mc_rtt.npz` (~1 MB) onto the cluster so every
job can skip the DANDI download.

## Phase 2 — smoke test (~15 min)

```bash
./cluster/local_drive.sh smoke
./cluster/local_drive.sh status   # watch until SMOKE TEST OK
```

`cluster/sbatch/smoke.sbatch` runs nvidia-smi → container pull → pytest →
a 1-fold / 1-seed / 1-budget block_cv. If it ends with `SMOKE TEST OK` in
`cluster_logs/smoke-<jobid>.out`, the path is verified for ~0.3 H100-hours
of budget.

If the smoke test fails, do not submit the real jobs: read the log, fix,
re-push, and re-smoke.

## Phase 3 — Round 1 batch (3 jobs, parallel, ~2 h wall time)

```bash
./cluster/local_drive.sh submit
```

Submits in parallel:

| sbatch | Wall time | Output |
|---|---|---|
| `block_cv_grid.sbatch` | ~75–90 min | `results/cluster/block_cv/block_cv.json` (streamed) |
| `perm_1000.sbatch` | ~90–150 min | `results/cluster/snn/permutation_test.json` |
| `hidden_dim_big.sbatch` | ~30–60 min | `results/cluster/hidden_dim_sweep/h{1024,2048}_results.json` |

Total budget: roughly 2.5–3.5 H100-hours (each job runs on one GPU,
concurrently). Poll every 15 min:

```bash
./cluster/local_drive.sh status
```

When `sacct` shows all three as `COMPLETED`, pull and aggregate:

```bash
./cluster/local_drive.sh pull
.venv/bin/python scripts/aggregate_cluster_results.py
```

The aggregator writes `results/cluster/summary.md` and an updated
`results/cluster/figures/headline_frontier_multiseed.png`.

## Phase 4 — review

After Round 1 lands, review:

1. The new `summary.md` and headline figure.
2. The remaining sources of variance or weakness.
3. Whether multi-seed block CV reveals cells where `trained_snn` beats,
   ties, or loses to `ridge_lag4`, and whether the permutation p-values
   support the decoding claim at every budget.

Based on the findings, queue Round 2 (designed but unsubmitted by default):

| sbatch | Purpose |
|---|---|
| `trained_snn_ensemble.sbatch` | 10-seed bagged trained SNN — usually an accuracy bump and a credible variance estimate |
| `ridge_lag_sweep.sbatch` | Shows lag-vs-R² plateaus at 4 (confirms the baseline is not cherry-picked) |
| `snn_sensitivity.sbatch` | Single-knob trained-SNN sensitivity across tau, threshold, k_history |

Round 2 budget: roughly 3.5–6 H100-hours total. Submit only cells whose
hypothesis is not already settled by Round 1.

## Phase 5 — decide whether the compute is exhausted

Re-run `scripts/aggregate_cluster_results.py` after Round 2, then check:

- Does the headline figure / table now have multi-seed error bars
  everywhere it reports a number?
- Is every "trained SNN ≈ ridge_lag4" claim backed by CI overlap or a
  tightly-bounded gap?
- Are the controls (shuffle, permutation null, sensitivity, lag sweep)
  all reported with enough rigor to survive review?
- Are there remaining single-seed numbers affecting a load-bearing claim?

If all four are settled, the compute has no further marginal value and the
numbers are final. Otherwise, design one more targeted job, run it,
re-aggregate, and iterate.

## Phase 6 — commit each iteration

After each `pull → aggregate`:

```bash
git add results/cluster/ cluster_logs/ 2>/dev/null
git status --short
git commit -m "cluster round N: <one-line summary of what changed>"
git push origin main
```

## Emergency stops

```bash
./cluster/local_drive.sh status                 # see what's running
./cluster/local_drive.sh shell                  # drop into the login pod
# from inside the pod:
scancel <jobid>                                 # kill one job
scancel -u $USER                                # kill ALL jobs (use sparingly)
sshare -u $USER                                 # remaining GPU-hour budget
```

If `sshare` shows the budget running low before Round 2, skip it: the
Round 1 results alone are enough to harden the headline claims.

## What never to touch

- Other users' login pods (`stanford/user=<not-you>`).
- `slurm-controller-*`, `slurm-accounting-*`, `mariadb-*`,
  `slurm-worker-*` infrastructure pods.
- The Slurm controller's ConfigMaps, partition definitions, RBAC, or
  ValidatingAdmissionPolicy.
- `helm upgrade` or `kubectl delete` on anything in `slurm`,
  `kube-system`, or other infrastructure namespaces.

`cluster/CLUSTER.md` and the cluster onboarding doc are the source of
truth on these guardrails.
