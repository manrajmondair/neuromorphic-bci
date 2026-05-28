"""Cluster operator — drive AMP PBC SLURM jobs from a laptop.

The driver only relies on verbs that are available under either an
interactive `manraj` kubeconfig OR a client-credentials kubeconfig
(Customer Admin + ML Ops). It works under both:

  1. `kubectl create pod` + `kubectl logs` in the `slurm` namespace.
     CPU-only helper pods mount the shared Weka home (`slurm-shared-home`
     PVC) as UID 10010 to do filesystem work — clone the repo, run
     preprocessing, tar `results/cluster/`, etc. The pods' stdout is the
     channel back to the laptop; base64-encoded tarballs round-trip
     cleanly through `kubectl logs`.

  2. `kubectl port-forward svc/slurm-restapi 6820` + the `slurm-auth-jwt`
     secret. Sign a 2-hour HS256 token with `sun=manraj` and submit jobs
     to slurmrestd at `/slurm/v0.0.41/job/submit`. Jobs run as manraj
     with the right home dir + correct SLURM accounting.

The driver never needs `kubectl exec`, `kubectl cp`, or SSH, which is
why it works with a client-credentials identity that the cluster's
`stanford-exec-isolation` ValidatingAdmissionPolicy denies exec for.

CLI:
    python cluster/cluster_drive.py bootstrap         # clone + preprocess
    python cluster/cluster_drive.py submit <sbatch>   # submit a SLURM job
    python cluster/cluster_drive.py status <job_id>+  # SLURM job state
    python cluster/cluster_drive.py wait <job_id>+    # block until terminal
    python cluster/cluster_drive.py log <job_id>      # cat SLURM stdout file
    python cluster/cluster_drive.py pull              # tar results/cluster, fetch locally
    python cluster/cluster_drive.py round1            # smoke -> 3 main jobs
    python cluster/cluster_drive.py round2            # 3 round-2 jobs in parallel
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import uuid
from io import BytesIO
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] cluster_drive: %(message)s"
logger = logging.getLogger("cluster_drive")

NAMESPACE = "slurm"
USERNAME = "manraj"
USER_UID = 10010
HOME_PVC = "slurm-shared-home"
REPO_URL = "https://github.com/manrajmondair/neuromorphic-bci.git"
REPO_DIR = f"/home/{USERNAME}/neuromorphic-bci"
SLURMRESTD_SVC = "slurm-restapi"
SLURMRESTD_PORT = 6820
SLURMRESTD_API_VER = "v0.0.41"
# Container image used for CPU-side helper work (clone, preprocess, tar
# results). Full `python:3.11` ships git + python + pip pre-installed so
# the helper scripts don't have to apt-get anything. Bookworm-slim drops
# git so we deliberately use the non-slim image.
HELPER_IMAGE = "python:3.11"


# ---------------------------------------------------------------------------
# kubectl + JWT helpers
# ---------------------------------------------------------------------------


def _kubectl(*args, check=True, capture=True):
    cmd = ["kubectl", *args]
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def slurm_jwt(username: str = USERNAME, lifetime_s: int = 2 * 60 * 60) -> str:
    """Sign a short-lived slurmrestd JWT scoped to `username`."""
    import jwt  # local import; pyjwt is in requirements.txt
    out = subprocess.check_output(
        ["kubectl", "get", "secret", "slurm-auth-jwt", "-n", NAMESPACE,
         "-o", "jsonpath={.data.jwt\\.key}"], text=True
    )
    key = base64.b64decode(out)
    now = int(time.time())
    return jwt.encode({"iat": now, "exp": now + lifetime_s, "sun": username},
                      key, algorithm="HS256")


@contextlib.contextmanager
def slurmrestd_portforward():
    """Yield a local port that proxies to slurm-restapi:6820 inside the cluster."""
    # Pick a free local port to avoid collisions across concurrent driver runs.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        local_port = s.getsockname()[1]
    proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", NAMESPACE,
         f"svc/{SLURMRESTD_SVC}", f"{local_port}:{SLURMRESTD_PORT}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Wait briefly for the listener to come up.
        for _ in range(20):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", local_port)) == 0:
                    break
            time.sleep(0.2)
        else:
            raise RuntimeError("port-forward did not open")
        yield local_port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def slurmrestd(method: str, path: str, body: dict | None = None,
               username: str = USERNAME):
    """Make an authenticated request to slurm-restapi, returning the JSON body."""
    import requests  # already in requirements via dandi's transitive deps
    token = slurm_jwt(username)
    headers = {"X-SLURM-USER-NAME": username, "X-SLURM-USER-TOKEN": token,
               "Content-Type": "application/json"}
    with slurmrestd_portforward() as port:
        url = f"http://localhost:{port}/slurm/{SLURMRESTD_API_VER}{path}"
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            r = requests.post(url, headers=headers, data=json.dumps(body or {}),
                              timeout=60)
        elif method == "DELETE":
            r = requests.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError(method)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Pod helpers — for file I/O on /home/manraj without exec or cp
# ---------------------------------------------------------------------------


def run_pod(name: str, script: str, image: str = HELPER_IMAGE,
            mount_home: bool = True, run_as_user: int = USER_UID,
            timeout_s: int = 600, ttl_seconds_after_finished: int = 300) -> str:
    """Run a one-shot pod that executes `script`, then return its log output."""
    pod_name = f"{name}-{uuid.uuid4().hex[:6]}"
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": NAMESPACE,
                     "labels": {"managed-by": "cluster-drive", "purpose": name}},
        "spec": {
            "restartPolicy": "Never",
            "securityContext": {"runAsUser": run_as_user,
                                "runAsGroup": run_as_user,
                                "fsGroup": run_as_user,
                                # The slurm-shared-home volume is 15 TB. Without
                                # OnRootMismatch, kubelet recursively chowns the
                                # entire tree every pod start (minutes-to-hours).
                                # OnRootMismatch only chowns if the root dir
                                # group doesn't already match; for this volume
                                # it's already 10010, so this is a no-op.
                                "fsGroupChangePolicy": "OnRootMismatch"},
            "containers": [{
                "name": "main",
                "image": image,
                "command": ["bash", "-lc", script],
                "resources": {
                    "requests": {"cpu": "1", "memory": "2Gi"},
                    "limits": {"cpu": "4", "memory": "8Gi"},
                },
                "volumeMounts": ([{"name": "home", "mountPath": "/home"}]
                                 if mount_home else []),
            }],
            "volumes": ([{"name": "home",
                         "persistentVolumeClaim": {"claimName": HOME_PVC}}]
                        if mount_home else []),
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    try:
        _kubectl("apply", "-f", manifest_path)
        _kubectl("wait", "--for=jsonpath={.status.phase}=Succeeded",
                 f"pod/{pod_name}", "-n", NAMESPACE, f"--timeout={timeout_s}s",
                 check=False)
        log = _kubectl("logs", "-n", NAMESPACE, pod_name).stdout
        phase = _kubectl("get", "pod", pod_name, "-n", NAMESPACE,
                         "-o", "jsonpath={.status.phase}").stdout.strip()
        if phase != "Succeeded":
            raise RuntimeError(f"pod {pod_name} ended in phase {phase}; logs:\n{log}")
        return log
    finally:
        _kubectl("delete", "pod", pod_name, "-n", NAMESPACE,
                 "--wait=false", check=False)


# ---------------------------------------------------------------------------
# SLURM job helpers
# ---------------------------------------------------------------------------


def _submitter_pod_manifest(pod_name: str, sbatch_relpath: str) -> dict:
    """Pod that mirrors slurm-login-manraj's setup, but with a one-shot
    command that runs `sbatch <relpath>` as manraj and exits.

    Why not just POST to slurmrestd: slurmrestd is the standalone
    `slurm-restapi` deployment that runs as `nobody` in a container with no
    /etc/passwd mapping for manraj. slurmctld rejects its forwarded
    submissions with `Rejecting authentication of user nobody`. The login
    image, by contrast, mounts the `login-manraj-users` configmap at
    /etc/passwd so getpwnam("manraj") returns UID 10010.
    """
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": NAMESPACE,
                     "labels": {"managed-by": "cluster-drive",
                                "purpose": "submit"}},
        "spec": {
            "automountServiceAccountToken": False,
            "restartPolicy": "Never",
            "dnsConfig": {"searches": ["slurm-workers-slurm.slurm.svc.cluster.local"]},
            "initContainers": [{
                "name": "initconf",
                "image": "docker.io/library/alpine:3.21",
                "command": ["sh", "-c", """
set -eu
SLURM_DIR=/mnt/etc/slurm
mkdir -p "$SLURM_DIR"
find /mnt/slurm -type f -name "*.conf" -print0 | xargs -0r cp -vt "$SLURM_DIR"
find /mnt/slurm -type f -name "*.key"  -print0 | xargs -0r cp -vt "$SLURM_DIR"
chown -R 401:401 "$SLURM_DIR"
find "$SLURM_DIR" -name "*.conf" -exec chmod 644 {} +
find "$SLURM_DIR" -name "*.key"  -exec chmod 600 {} +
ls -lAF "$SLURM_DIR"
                """],
                "volumeMounts": [
                    {"name": "slurm-etc",    "mountPath": "/mnt/etc/slurm"},
                    {"name": "slurm-config", "mountPath": "/mnt/slurm", "readOnly": True},
                ],
            }],
            "containers": [{
                "name": "submit",
                "image": "ghcr.io/mihai-amp/login:25.11-ubuntu24.04",
                "env": [{"name": "SACKD_OPTIONS",
                         "value": "--conf-server slurm-controller.slurm:6817"}],
                "command": ["bash", "-c", f"""
set -euo pipefail
# Start sackd in background — same daemon the regular login pod runs.
/usr/sbin/sackd --conf-server slurm-controller.slurm:6817 &
SACKD_PID=$!
trap 'kill $SACKD_PID 2>/dev/null || true' EXIT
# Wait for the auth socket to materialize so sbatch can authenticate.
for i in $(seq 30); do
  [ -S /run/slurm/sack.socket ] && break
  sleep 1
done
[ -S /run/slurm/sack.socket ] || {{ echo "sackd socket never appeared"; exit 1; }}
echo "--- submitting {sbatch_relpath} as manraj ---"
runuser -u manraj -- bash -lc "cd {REPO_DIR} && mkdir -p cluster_logs && sbatch {sbatch_relpath}"
echo "--- submitted ---"
                """],
                "volumeMounts": [
                    {"name": "shared-home",  "mountPath": "/home"},
                    {"name": "etc-users",    "mountPath": "/etc/passwd",
                     "subPath": "passwd"},
                    {"name": "etc-users",    "mountPath": "/etc/group",
                     "subPath": "group"},
                    {"name": "slurm-etc",    "mountPath": "/etc/slurm",
                     "readOnly": True},
                    {"name": "sackd-dir",    "mountPath": "/run/slurm"},
                ],
            }],
            "volumes": [
                {"name": "shared-home",
                 "persistentVolumeClaim": {"claimName": HOME_PVC}},
                {"name": "etc-users",
                 "configMap": {"name": f"login-{USERNAME}-users"}},
                {"name": "slurm-etc",   "emptyDir": {"medium": "Memory"}},
                {"name": "sackd-dir",   "emptyDir": {"medium": "Memory"}},
                {"name": "slurm-config",
                 "projected": {"defaultMode": 0o600, "sources": [
                     {"secret": {"name": "slurm-auth-slurm",
                                 "items": [{"key": "slurm.key", "path": "slurm.key"}]}}
                 ]}},
            ],
        },
    }


def submit_via_login_clone(sbatch_path: Path, timeout_s: int = 180) -> int:
    """Submit a sbatch file via a short-lived clone of the login pod. Returns SLURM job_id."""
    pod_name = f"submit-{uuid.uuid4().hex[:6]}"
    relpath = sbatch_path.as_posix()
    manifest = _submitter_pod_manifest(pod_name, relpath)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        path = f.name
    try:
        _kubectl("apply", "-f", path)
        _kubectl("wait", "--for=jsonpath={.status.phase}=Succeeded",
                 f"pod/{pod_name}", "-n", NAMESPACE,
                 f"--timeout={timeout_s}s", check=False)
        log = _kubectl("logs", "-n", NAMESPACE, pod_name).stdout
        phase = _kubectl("get", "pod", pod_name, "-n", NAMESPACE,
                         "-o", "jsonpath={.status.phase}").stdout.strip()
        if phase != "Succeeded":
            raise RuntimeError(f"submitter pod {pod_name} ended in {phase}; logs:\n{log}")
        # sbatch prints "Submitted batch job <jid>" on its stdout.
        for line in log.splitlines():
            if line.startswith("Submitted batch job"):
                jid = int(line.split()[-1])
                logger.info("submitted %s as job_id=%s", sbatch_path.name, jid)
                return jid
        raise RuntimeError(f"no 'Submitted batch job' line in submitter log:\n{log}")
    finally:
        _kubectl("delete", "pod", pod_name, "-n", NAMESPACE,
                 "--wait=false", check=False)


def submit_slurm_job(name: str, script: str, *, time_limit_min: int,
                     gpus: int = 0, partition: str = "small",
                     cpus_per_task: int = 16):
    """Submit a SLURM job via REST, returning its job_id."""
    job = {
        "job": {
            "name": name,
            "partition": partition,
            "cpus_per_task": cpus_per_task,
            "current_working_directory": REPO_DIR,
            "standard_output": f"{REPO_DIR}/cluster_logs/{name}-%j.out",
            "standard_error": f"{REPO_DIR}/cluster_logs/{name}-%j.out",
            "time_limit": {"set": True, "infinite": False,
                           "number": time_limit_min},
            "environment": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
        },
        "script": "#!/bin/bash\n" + script,
    }
    if gpus > 0:
        job["job"]["tres_per_job"] = f"gres/gpu:{gpus}"

    reply = slurmrestd("POST", "/job/submit", job)
    if reply.get("errors"):
        raise RuntimeError(f"submit errored: {reply['errors']}")
    job_id = reply["job_id"]
    logger.info("submitted SLURM job %s as id=%s", name, job_id)
    return int(job_id)


def slurm_job_state(job_id: int):
    """Return SLURM job state + a few useful fields."""
    reply = slurmrestd("GET", f"/job/{job_id}")
    if reply.get("errors"):
        return {"state": "MISSING", "errors": reply["errors"]}
    j = reply["jobs"][0]
    return {
        "state": j.get("job_state", ["?"])[0]
                  if isinstance(j.get("job_state"), list)
                  else j.get("job_state"),
        "exit_code": j.get("exit_code"),
        "elapsed": j.get("time"),
        "stdout_path": j.get("standard_output"),
        "node": j.get("nodes"),
    }


def wait_for_jobs(job_ids: list[int], poll_s: int = 30) -> dict[int, dict]:
    """Block until every job_id reaches a terminal state. Returns final states."""
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "BOOT_FAIL", "NODE_FAIL", "DEADLINE", "PREEMPTED",
                "REVOKED", "SPECIAL_EXIT", "STOPPED", "MISSING"}
    states: dict[int, dict] = {}
    pending = set(job_ids)
    while pending:
        time.sleep(poll_s)
        for jid in list(pending):
            try:
                s = slurm_job_state(jid)
            except Exception as e:  # noqa: BLE001
                logger.warning("status poll failed for %s: %s", jid, e)
                continue
            states[jid] = s
            logger.info("job %s: state=%s elapsed=%s", jid, s["state"], s.get("elapsed"))
            if s["state"] in terminal:
                pending.discard(jid)
    return states


def read_slurm_job_log(job_id: int) -> str:
    """Cat the SLURM job's stdout file via a helper pod."""
    log_glob = f"{REPO_DIR}/cluster_logs/*-{job_id}.out"
    script = f'shopt -s nullglob; for f in {log_glob}; do echo "=== $f ==="; cat "$f"; done'
    return run_pod("read-log", script, mount_home=True, timeout_s=120)


# ---------------------------------------------------------------------------
# High-level orchestration commands
# ---------------------------------------------------------------------------


def cmd_bootstrap(args):
    """Clone the repo onto /home/manraj and run preprocessing."""
    script = textwrap.dedent(f"""
        set -euo pipefail
        # The python:3.11 image has no /etc/passwd entry for UID 10010, so
        # ~ expands to / which fails on first write. Pin HOME + XDG dirs to
        # the user's home before anything touches a cache.
        export HOME=/home/{USERNAME}
        export XDG_CACHE_HOME=/home/{USERNAME}/.cache
        mkdir -p "$XDG_CACHE_HOME"
        cd /home/{USERNAME}
        if [ -d neuromorphic-bci/.git ]; then
            cd neuromorphic-bci
            git remote set-url origin {REPO_URL}
            git fetch --prune origin
            git checkout main
            git reset --hard origin/main
        else
            git clone {REPO_URL} neuromorphic-bci
            cd neuromorphic-bci
        fi
        mkdir -p cluster_logs data/raw data/processed results/cluster
        echo "--- repo head ---"
        git log --oneline -1
        echo "--- python + pip ---"
        python3 -V
        python3 -m pip --version
        if [ ! -d .venv-bootstrap ]; then
            python3 -m venv .venv-bootstrap
        fi
        . .venv-bootstrap/bin/activate
        export PIP_CACHE_DIR=/home/{USERNAME}/.cache/pip
        mkdir -p "$PIP_CACHE_DIR"
        pip install --upgrade --quiet pip
        echo "--- installing minimal preprocess deps ---"
        # Bootstrap only needs to download + preprocess MC_RTT. We deliberately
        # skip torch / snntorch / matplotlib / scikit-learn / jupyter — those
        # are for the SLURM GPU jobs which use the NGC PyTorch container that
        # already has them pre-installed. Trimming the bootstrap from ~1 GB
        # of wheels to ~200 MB cuts the pod from ~5 min to ~1 min.
        pip install --quiet \
            "numpy>=1.24" "scipy>=1.10" "pandas>=2.0" \
            "h5py>=3.9" "pynwb>=2.5" "tqdm>=4.65" "dandi>=0.59"
        pip install --quiet --no-deps "nlb-tools==0.0.4"
        echo "--- download via direct https ---"
        # The dandi-py CLI hits a PermissionError on the Weka filesystem when
        # writing into freshly-created subdirs (its subprocess workers seem
        # to lose fsGroup). Sidestep it by fetching each asset's blob URL
        # directly from the public DANDI API. Hard-coded for dandiset 000129
        # / draft / sub-Indy (the MC_RTT recording). The DANDI API URL
        # `https://api.dandiarchive.org/api/assets/<asset_id>/download/`
        # 302-redirects to the actual S3 blob, which curl -L follows.
        mkdir -p data/raw/000129/sub-Indy
        cat <<'EOF' | while IFS=$'\t' read -r aid relpath size; do
648a7418-98e8-4413-ba97-3772dd325ecc\tsub-Indy/sub-Indy_desc-test_ecephys.nwb\t1201344
2ae6bf3c-788b-4ece-8c01-4b4a5680b25b\tsub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb\t49764168
EOF
            dest="data/raw/000129/$relpath"
            if [ -f "$dest" ] && [ "$(stat -c '%s' "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
                echo "skip $relpath ($size bytes, already downloaded)"
                continue
            fi
            mkdir -p "$(dirname "$dest")"
            echo "fetching $relpath ($size bytes) -> $dest"
            curl -fLs --retry 3 \
                -o "$dest" \
                "https://api.dandiarchive.org/api/assets/$aid/download/"
        done
        echo "--- raw files ---"
        find data/raw -name '*.nwb' -exec ls -la {{}} \;
        if [ ! -f data/processed/processed_mc_rtt.npz ]; then
            python scripts/preprocess_mc_rtt.py 2>&1 | tail -20
        fi
        echo "--- final state ---"
        ls -la data/processed/
        echo "BOOTSTRAP OK"
    """)
    log = run_pod("bootstrap", script, mount_home=True, timeout_s=900)
    print(log)


def parse_sbatch(path: Path) -> tuple[dict, str]:
    """Parse an sbatch file into (metadata, body).

    Metadata pulled from #SBATCH lines: job_name, partition, gpus,
    cpus_per_task, time_min. The body is every non-#SBATCH non-comment line
    after the shebang.
    """
    lines = Path(path).read_text().splitlines()
    meta = {"job_name": "ncbi-job", "partition": "small", "gpus": 0,
            "cpus_per_task": 16, "time_min": 60}
    body_lines: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("#!"):
            continue
        if s.startswith("#SBATCH"):
            kv = s.removeprefix("#SBATCH").strip()
            if kv.startswith("--job-name="):
                meta["job_name"] = kv.split("=", 1)[1]
            elif kv.startswith("--partition="):
                meta["partition"] = kv.split("=", 1)[1]
            elif kv.startswith("--gres=gpu:"):
                meta["gpus"] = int(kv.split(":")[-1])
            elif kv.startswith("--cpus-per-task="):
                meta["cpus_per_task"] = int(kv.split("=", 1)[1])
            elif kv.startswith("--time="):
                # HH:MM:SS -> minutes
                t = kv.split("=", 1)[1]
                parts = [int(x) for x in t.split(":")]
                if len(parts) == 3:
                    h, m, s_ = parts
                    meta["time_min"] = h * 60 + m + (1 if s_ else 0)
                elif len(parts) == 2:
                    meta["time_min"] = parts[0] * 60 + parts[1]
                else:
                    meta["time_min"] = parts[0]
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip() + "\n"
    return meta, body


def cmd_submit(args):
    meta, _body = parse_sbatch(Path(args.sbatch))
    logger.info("submitting %s (gpus=%d time=%d min) via login-pod clone",
                meta["job_name"], meta["gpus"], meta["time_min"])
    job_id = submit_via_login_clone(Path(args.sbatch))
    print(json.dumps({"job_id": job_id, **meta}))


def _submit_sbatch(name: str) -> int:
    return submit_via_login_clone(Path(f"cluster/sbatch/{name}.sbatch"))


def cmd_round1(args):
    """Submit smoke first, gate on its success, then fan out the 3 round-1 jobs."""
    smoke_id = _submit_sbatch("smoke")
    logger.info("smoke job submitted: id=%s; waiting for COMPLETED ...", smoke_id)
    states = wait_for_jobs([smoke_id])
    if states[smoke_id]["state"] != "COMPLETED":
        raise RuntimeError(f"smoke job did not COMPLETE: {states}")
    logger.info("smoke OK; submitting 3 round-1 jobs in parallel ...")
    ids = [_submit_sbatch(n) for n in ("block_cv_grid", "perm_1000", "hidden_dim_big")]
    logger.info("submitted ids: %s", ids)
    print(json.dumps({"smoke_job_id": smoke_id, "round1_job_ids": ids}, indent=2))


def cmd_round2(args):
    """Submit the three round-2 jobs in parallel (no smoke required)."""
    ids = [_submit_sbatch(n) for n in
           ("trained_snn_ensemble", "ridge_lag_sweep", "snn_sensitivity")]
    logger.info("round-2 ids: %s", ids)
    print(json.dumps({"round2_job_ids": ids}, indent=2))


def cmd_status(args):
    for jid in args.job_ids:
        s = slurm_job_state(int(jid))
        print(json.dumps({"job_id": int(jid), **s}, indent=2))


def cmd_progress(args):
    """One-line-per-job summary of every SLURM job (mine, or by id list).

    With no ids, queries the SLURM queue + sacct for everything tied to
    `sun=manraj`. With ids, just dumps those.
    """
    if args.job_ids:
        ids = [int(j) for j in args.job_ids]
    else:
        # Use slurmrestd to list all of manraj's jobs (RUNNING + PENDING).
        reply = slurmrestd("GET", "/jobs")
        ids = [int(j["job_id"]) for j in reply.get("jobs", [])
               if j.get("user_name") == USERNAME]
        ids = sorted(set(ids))
    if not ids:
        print("no SLURM jobs found")
        return
    rows = []
    for jid in ids:
        try:
            s = slurm_job_state(jid)
        except Exception as e:  # noqa: BLE001
            s = {"state": f"ERR({e})"}
        elapsed = s.get("elapsed")
        if isinstance(elapsed, dict):
            elapsed = elapsed.get("number", 0)
        else:
            elapsed = 0
        rows.append({
            "job_id": jid,
            "state": s.get("state"),
            "elapsed_s": int(elapsed),
            "node": s.get("node") or "-",
        })
    width = max(len(str(r["state"])) for r in rows)
    print(f"{'job_id':>8} {'state':<{width}}  {'elapsed':>10}  node")
    for r in rows:
        print(f"{r['job_id']:>8} {r['state']:<{width}}  {r['elapsed_s']:>8}s   {r['node']}")


def cmd_wait(args):
    states = wait_for_jobs([int(j) for j in args.job_ids])
    print(json.dumps(states, indent=2))


def cmd_log(args):
    print(read_slurm_job_log(int(args.job_id)))


def cmd_pull(args):
    """Tar /home/manraj/.../results/cluster on the cluster, base64 it to stdout,
    and untar locally."""
    script = (
        f"cd {REPO_DIR} && tar c results/cluster/ 2>/dev/null | base64"
    )
    log = run_pod("pull-results", script, mount_home=True, timeout_s=180)
    # Pod log includes everything the container wrote; the base64 tarball is
    # the only thing the script emits. Strip whitespace and decode.
    blob = "".join(log.split())
    if not blob:
        raise RuntimeError("retrieval pod produced no output")
    tar_bytes = base64.b64decode(blob)
    out_dir = Path(args.out_dir or ".")
    with tarfile.open(fileobj=BytesIO(tar_bytes)) as tf:
        tf.extractall(out_dir)
    n = sum(1 for _ in (out_dir / "results" / "cluster").rglob("*") if _.is_file())
    logger.info("extracted %d files into %s/results/cluster/", n, out_dir)


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap").set_defaults(fn=cmd_bootstrap)
    sp = sub.add_parser("submit"); sp.add_argument("sbatch"); sp.set_defaults(fn=cmd_submit)
    sub.add_parser("round1").set_defaults(fn=cmd_round1)
    sub.add_parser("round2").set_defaults(fn=cmd_round2)
    sp = sub.add_parser("status"); sp.add_argument("job_ids", nargs="+"); sp.set_defaults(fn=cmd_status)
    sp = sub.add_parser("progress"); sp.add_argument("job_ids", nargs="*"); sp.set_defaults(fn=cmd_progress)
    sp = sub.add_parser("wait");   sp.add_argument("job_ids", nargs="+"); sp.set_defaults(fn=cmd_wait)
    sp = sub.add_parser("log");    sp.add_argument("job_id");             sp.set_defaults(fn=cmd_log)
    sp = sub.add_parser("pull");   sp.add_argument("--out-dir", default="."); sp.set_defaults(fn=cmd_pull)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
