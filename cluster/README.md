# Cluster operator's manual

Everything in this directory targets the AMP PBC SLURM cluster (32× H100, single-node, container-based). Use this only after the one-time setup in `cluster/SETUP.md` is complete.

## Quick map

| File | Purpose |
|---|---|
| `sbatch/smoke.sbatch` | 30-min smoke test: nvidia-smi + container pull + pytest + tiny block_cv. Run this first. |
| `sbatch/block_cv_grid.sbatch` | Block CV × 10 seeds × 4 budgets × 4 folds × 3 models. ~90 min. |
| `sbatch/perm_1000.sbatch` | Permutation test, n_perm=1000 across 4 budgets. ~120 min. |
| `sbatch/hidden_dim_big.sbatch` | hidden_dim ∈ {1024, 2048} × 3 seeds × 2 budgets. ~45 min. |
| `local_drive.sh` | Helper run from a laptop: stages repo + data, submits jobs, pulls results. |

## What gets written where on the cluster

Inside `/home/manraj/neuromorphic-bci/`:

- `data/processed/processed_mc_rtt.npz` — pre-staged from laptop, reused by all jobs.
- `results/cluster/block_cv/block_cv.json` — block-CV grid output (streamed as cells finish).
- `results/cluster/snn/permutation_test.json` — perm test output.
- `results/cluster/hidden_dim_sweep/{h1024,h2048}_results.json` — hidden-dim sweep output.
- `results/cluster/figures/*.png` — any figures the scripts emit.
- `cluster_logs/<jobname>-<jobid>.out` — stdout+stderr per job.

The `results/cluster/` namespace is deliberately separate from the existing `results/` so we never overwrite the canonical T4-era numbers. The local pull script merges them back into `results/cluster/` on the laptop.

## Running a job

From the login pod:

```bash
sbatch cluster/sbatch/smoke.sbatch       # verify path works
squeue -u $USER                          # see queue
sacct -u $USER -S today                  # job history today
scancel <jobid>                          # kill stuck job
```

Submit the three real jobs in parallel:

```bash
sbatch cluster/sbatch/block_cv_grid.sbatch
sbatch cluster/sbatch/perm_1000.sbatch
sbatch cluster/sbatch/hidden_dim_big.sbatch
```

They each request 1 GPU on `small`; if the partition has capacity they all run concurrently and the total wall is `max(individual wall times)` ≈ 2 hours.

## Reading stream output

The block-CV script writes `block_cv.json` after every (budget, fold, seed) cell so you can `cat` it mid-run to see partial progress. The other scripts emit progress through their `cluster_logs/<job>-<jobid>.out` files.

```bash
tail -f cluster_logs/block_cv-<jobid>.out          # live log
jq '.rows | length' results/cluster/block_cv/block_cv.json   # cells finished so far
```

## Pulling results back to the laptop

After all three jobs finish (`sacct -u $USER -S today` shows `COMPLETED`):

```bash
# Run from your laptop, NOT inside the pod
./cluster/local_drive.sh pull
```

This `kubectl cp`s the entire `results/cluster/` tree back onto the laptop, where it is merged into the local working tree before the final figures are regenerated.

## Things that will bite you

- **Don't pass `--cpus-per-task` lower than 16 on container jobs.** First-time enroot squashfs build is single-threaded; with 2 CPUs it takes ~30 min, with 16 it takes ~3 min.
- **Don't use `docker.io#library/...` image refs.** Bare names (`alpine:latest`) or NGC URIs (`nvcr.io#...`) only. We use the NGC PyTorch image.
- **Don't run anything compute-heavy in the login pod.** It has no GPU and very small CPU limits. Always sbatch.
- **Don't kill `slurm-*` infrastructure pods.** See `cluster/CLUSTER.md` for the full safety list.

## Budget hygiene

```bash
sshare -u $USER       # remaining GPU-hour budget
```

The three real jobs together consume roughly **2–3 H100 hours**. Smoke test consumes ~0.3 H100 hours. Check `sshare` before submitting a sweep.

