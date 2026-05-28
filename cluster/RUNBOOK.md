# Cluster runbook (what happens after kubeconfig lands)

This is the playbook I follow from the moment you paste me the output of `kubectl auth whoami` until the iteration loop converges.

## Phase 0 — verify connectivity (10 s)

```bash
kubectl get pod -n slurm -l stanford/user=manraj
```

Expect one pod whose name starts with `slurm-login-manraj-`. If empty, ping Anthony — the LoginSet hasn't been provisioned yet.

## Phase 1 — stage data + repo on the cluster (1–2 min)

```bash
./cluster/local_drive.sh push
```

Clones `https://github.com/manrajmondair/neuromorphic-bci.git` into `/home/manraj/neuromorphic-bci` on the cluster (or `git pull --ff-only` if it already exists), then `kubectl cp`s the local `data/processed/processed_mc_rtt.npz` (1 MB) onto the cluster so every job can skip the DANDI download.

## Phase 2 — smoke test (~15 min)

```bash
./cluster/local_drive.sh smoke
./cluster/local_drive.sh status   # watch until SMOKE TEST OK
```

The `cluster/sbatch/smoke.sbatch` job: nvidia-smi → container pull → pytest → a 1-fold / 1-seed / 1-budget block_cv. If it ends with `SMOKE TEST OK` in `cluster_logs/smoke-<jobid>.out`, the path is verified for ~0.3 H100-hours of budget.

If smoke fails: do NOT submit the real jobs. Read the log, fix, re-push, re-smoke.

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

Total budget consumed: roughly 2.5–3.5 H100-hours (each runs on one GPU but concurrently). Poll every 15 min:

```bash
./cluster/local_drive.sh status
```

When `sacct` shows all three as `COMPLETED`, pull:

```bash
./cluster/local_drive.sh pull
.venv/bin/python scripts/aggregate_cluster_results.py
```

The aggregator writes `results/cluster/summary.md` and an updated `results/cluster/figures/headline_frontier_multiseed.png`. Commit both immediately so the public repo carries the new state.

## Phase 4 — Step back and review

After Round 1 lands I read:

1. The new summary.md and headline figure.
2. Sources of remaining variance / weakness.
3. Specifically: did multi-seed block_cv reveal cells where trained_snn beats / ties / loses to ridge_lag4? Does the permutation p-value support the order-matters story at every budget?

Based on what we find, I queue Round 2 (designed but unsubmitted by default):

| sbatch | Purpose |
|---|---|
| `trained_snn_ensemble.sbatch` | 10-seed bagged trained SNN — usually gives an accuracy bump and a credible variance |
| `ridge_lag_sweep.sbatch` | Show lag-vs-R² plateaus at 4 (confirms baseline isn't cherry-picked) |
| `snn_sensitivity.sbatch` | Single-knob trained-SNN sensitivity across tau, threshold, k_history |

Round 2 budget: roughly 3.5–6 H100-hours total. Only submit cells whose hypothesis isn't already settled by Round 1.

## Phase 5 — Decide whether the compute is exhausted

Re-run `scripts/aggregate_cluster_results.py` after Round 2. Then ask:

- Does the headline figure / table now have multi-seed error bars everywhere it claimed a number?
- Is every "trained SNN ≈ ridge_lag4" claim now backed by either CI overlap or a tightly-bounded gap?
- Are the controls (shuffle, permutation null, sensitivity, lag sweep) all reported with enough rigor to survive review?
- Are there remaining single-seed numbers in the paper that affect a load-bearing claim?

If NO to all four: the compute is of no further marginal value and we stop. The paper has its numbers.

If any YES: design one more targeted job, run it, re-aggregate, iterate.

## Phase 6 — Commit + push every iteration

After each round of `pull → aggregate`:

```bash
git add results/cluster/ cluster_logs/ 2>/dev/null
git status --short
git commit -m "cluster round N: <one-line summary of what changed>"
git push origin main
```

Per the standing "push every iteration" rule, the GitHub repo always reflects the latest cluster state within ~30 min of new results landing.

## Emergency stops

```bash
./cluster/local_drive.sh status                 # see what's running
./cluster/local_drive.sh shell                  # drop into the login pod
# from inside the pod:
scancel <jobid>                                 # kill one job
scancel -u $USER                                # kill ALL my jobs (use sparingly)
sshare -u $USER                                 # remaining GPU-hour budget
```

If `sshare` shows the budget running low BEFORE Round 2: skip Round 2 and just commit the Round 1 results — they alone are enough to harden the paper's headline claims.

## What I never touch

- Other students' login pods (`stanford/user=<not-manraj>`).
- `slurm-controller-*`, `slurm-accounting-*`, `mariadb-*`, `slurm-worker-*` infrastructure pods.
- The Slurm controller's ConfigMaps, partition definitions, RBAC, or ValidatingAdmissionPolicy.
- `helm upgrade` or `kubectl delete` on anything in `slurm`, `kube-system`, or other infra namespaces.

CLAUDE.md at the repo root and the AMP onboarding doc are the source of truth on these guardrails.
