# Cluster setup — what you (Manraj) need to do manually

The Omniva CLI's first-time OAuth requires a browser and a sudo password, neither of which my session can drive. Once these five steps are done, I can take over the rest from `kubectl` here.

## Steps for you to run on your laptop

### 1. Download the `om` binary

Follow [Omniva prerequisites](https://docs.omniva.com/user/getting-started/access/#prerequisites) and download the `om` binary into `~/Downloads/`.

### 2. Install it into your PATH

macOS blocks direct moves from Downloads to `/usr/local/bin`, so copy through `/private/tmp` first:

```bash
chmod +x ~/Downloads/om
cp ~/Downloads/om /private/tmp/om
chmod +x /private/tmp/om
sudo mv /private/tmp/om /usr/local/bin/om
om --version
```

### 3. Log in via browser OAuth

```bash
om login
```

This opens a browser. Sign in with your AMP-invited account; the CLI captures the token automatically.

### 4. Generate your kubeconfig

```bash
om create kubeconfig --k8s-cluster amp-internal
```

### 5. Verify your username

```bash
kubectl auth whoami
```

This MUST print a line like `Username: manraj`. If the username is anything other than `manraj`, paste me the output before doing anything else — I'll patch `CLAUDE.md` and every sbatch script with the correct value. Likely names if the convention surprises us: `mmondair`, `msmondair`, `manrajm`, `manraj-mondair`.

### 6. Locate your login pod (sanity check)

```bash
kubectl get pod -n slurm -l stanford/user=manraj
```

Should print one pod whose name starts with `slurm-login-manraj-`. Paste me the pod name once you see it; I'll use it in the cluster-driver scripts.

## When all six are done

Reply to me with:

1. The output of `kubectl auth whoami` (one line is enough).
2. The pod name from step 6.

Then I take over — push the repo + processed data onto the cluster, submit the three sbatch jobs, monitor, pull results, and regenerate figures.
