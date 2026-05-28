# Compute environment reference

The heavy multi-seed grids in this project were produced on a shared
32× H100 SLURM cluster (4 nodes × 8× H100 80 GB HBM3, single-node jobs,
container-based). This file documents that environment, the image and
budget policy, and the operational guardrails. It is the reference the
job scripts in this directory assume; complete the one-time access setup
in `SETUP.md` first.

## Environment

- **Hardware**: 32× H100 (4 nodes × 8 GPUs, 80 GB HBM3 each).
- **Scheduler**: SLURM, single-node partitions only (`MaxNodes=1`).
- **Access**: container-based exec into a per-user login pod; no SSH and
  no public port-forwarding. File movement is over the network (HTTPS,
  S3, Hugging Face Hub) or `kubectl cp`.
- **Home**: a shared Weka filesystem mounted on the login pod and all
  workers (soft cap ~1 TB per user — delete or move stale checkpoints).

The login pod is a small CPU shell with no GPU. Never run training,
evaluation, or container pulls there — all compute goes through `sbatch`.

## Partitions

| Partition | Max walltime | Use for |
|---|---|---|
| `small`  | 24 h | single-GPU jobs, quick experiments (default) |
| `medium` | 5 d  | multi-GPU runs, longer training |
| `big`    | 5 d  | large reserved slots — restricted access |

## GPU-hour budget

A per-user GPU-hour cap is enforced by SLURM QoS; once it is hit, new
jobs queue until the budget resets.

```bash
sshare -u $USER                                    # remaining budget
sacctmgr show qos qos-$USER format=GrpTRESMins     # absolute cap (minutes)
```

Prefer fewer, larger jobs over many small ones, default to
`--gres=gpu:1` unless multi-GPU is genuinely required, and check
feasibility against the remaining budget before launching a sweep.

## Containers

GPU work runs inside the NGC PyTorch image, which ships CUDA, NCCL, and
cuDNN matched to the host drivers:

```bash
srun --gres=gpu:1 --cpus-per-task=16 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  python -c "import torch; print(torch.cuda.device_count())"   # prints 1
```

Two rules that save time:

- **Always pass `--cpus-per-task=16` on container jobs.** The first use
  of an image on a worker builds a squashfs (~20 GB for PyTorch). That
  build is single-threaded and uses only the CPUs the job was given —
  ~30 min with the default 2 CPUs, ~3 min with 16. The squashfs is then
  cached on that worker.
- **Image reference syntax.** Use bare names for Docker Hub
  (`python:3.12-slim`) and the `<registry>#<path>` form for everything
  else (`nvcr.io#nvidia/pytorch:24.12-py3`). The specific
  `docker.io#library/<name>` form breaks the manifest pipeline.

Inside the container, install this project's runtime with:

```bash
pip install --no-deps nlb-tools && pip install -r requirements.txt
```

## Storage layout

- `/home/<user>/neuromorphic-bci` — working copy of this repo on the cluster.
- `data/processed/processed_mc_rtt.npz` — staged once from a laptop (~1 MB)
  and reused by every job, so workers skip the DANDI download.
- `results/cluster/<experiment>/…` — all cluster outputs land here, kept
  deliberately separate from the local `results/` tree so the two never
  collide.

## Guardrails

- Only `sbatch` compute; never run heavy work in the login pod.
- Single-node only — no multi-node DDP. Use 1–8 GPUs on one node.
- Do not touch other users' login pods or the cluster's infrastructure
  pods (`slurm-controller-*`, `slurm-accounting-*`, `mariadb-*`,
  `slurm-worker-*`), ConfigMaps, partition definitions, RBAC, or admission
  policies.
- All actions on the cluster are audit-logged; do not probe other
  namespaces or other users' pods.
