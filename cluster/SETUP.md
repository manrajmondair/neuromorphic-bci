# Cluster setup — one-time

The Omniva CLI's first-time OAuth requires a browser and a sudo password. Once these steps are done, every subsequent cluster operation can be automated through `cluster/cluster_drive.py`.

## Steps

### 1. Download the `om` binary

Follow [Omniva prerequisites](https://docs.omniva.com/user/getting-started/access/#prerequisites) and download the `om` binary into `~/Downloads/`.

### 2. Install it into your PATH

macOS blocks direct moves from Downloads to `/usr/local/bin`, so copy through `/private/tmp` first:

```bash
chmod +x ~/Downloads/om
sudo install -m 0755 ~/Downloads/om /usr/local/bin/om
om --version
```

### 3. Log in

Either flow works:

```bash
om login                                 # interactive — opens a browser, signs in as your Stanford account
# OR
om login --client-credentials-file creds.json    # M2M flow, headless; creds.json contains client_id + client_secret
```

Interactive login gives a kubeconfig that can `kubectl exec` into your own login pod directly. The M2M client identity (Customer Admin role) cannot exec into login pods because of the cluster's `ValidatingAdmissionPolicy`, but it has every other right the driver needs: read secrets, port-forward, create pods, read pod logs.

### 4. Generate the kubeconfig

```bash
om create kubeconfig --k8s-cluster amp-internal
kubectl auth whoami
kubectl get pod -n slurm -l stanford/user=manraj
```

`whoami` should print `Username: manraj` (interactive flow) or a `bot-*` identity (M2M flow). Either is fine — the driver auto-detects which path to use.

## After setup

Drive everything from this laptop via:

```bash
python cluster/cluster_drive.py bootstrap      # clone + preprocess on the cluster
python cluster/cluster_drive.py round1         # smoke -> 3 main jobs
python cluster/cluster_drive.py status <jid>
python cluster/cluster_drive.py wait <jid>+
python cluster/cluster_drive.py log <jid>
python cluster/cluster_drive.py pull           # tar results/cluster -> back to laptop
```

See `cluster/RUNBOOK.md` for the full playbook from setup to results.
