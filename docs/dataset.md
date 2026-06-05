# Dataset: Neural Latents Benchmark MC_RTT

This is the single source of truth for what dataset we're using and how to
get it. Both branches assume this file. Do not duplicate dataset info
elsewhere — link here.

## What it is

**Neural Latents Benchmark (NLB) — MC_RTT.** Spiking activity recorded
from primary motor cortex during a self-paced random target reaching task,
along with simultaneously recorded cursor position, finger position, and
target position. This is the canonical reaching dataset used in the NLB
suite, originally collected by Makin, O'Doherty, Cardoso de Oliveira, and
Sabes.

This dataset is the right fit for the project because it contains real
motor-cortex activity during continuous movement — the decoding target
(2D cursor velocity) is computed from the cursor position trace.

## Distribution

The dataset lives on DANDI as **dandiset 000129** (`NLB_MC_RTT`).
The NLB project provides loader utilities in the `nlb_tools` Python
package; we depend on it via `requirements.txt`.

| Resource | Where |
|---|---|
| DANDI dandiset | `DANDI:000129` |
| NLB tools | https://github.com/neurallatents/nlb_tools |
| NLB paper | Pei et al., NeurIPS 2021 (`refs.bib :: nlb2021`) |

The exact dandiset version pin lives in `scripts/download_mc_rtt.py` as
`DANDISET_ID` and `DANDISET_VERSION` — change in one place if needed.

## How to get it

From the repo root, with the project venv activated:

```bash
pip install -r requirements.txt
pip install --no-deps nlb-tools           # upstream pin clash on pandas — see below
python scripts/download_mc_rtt.py
```

`nlb_tools` is installed separately with `--no-deps` because its setup
pins `pandas<=1.3.4` even though only basic DataFrame ops are used. We
need a modern pandas, so we let pip resolve everything else first and
then drop `nlb_tools` in on top.

The download writes NWB file(s) into `data/raw/`. The directory is
gitignored, so each contributor downloads locally and the bytes never
hit the repo.

To verify the download worked:

```bash
ls data/raw/000129/    # should contain at least one .nwb file
```

## Layout once downloaded

```
data/raw/
└── 000129/
    └── sub-Indy/
        └── sub-Indy_desc-train_behavior+ecephys.nwb     # exact name set by DANDI
```

## Going from raw to processed

```bash
python scripts/preprocess_mc_rtt.py
# writes data/processed/processed_mc_rtt.npz, schema in docs/data_interface.md
```

After this, **both branches** load the dataset the same way:

```python
from src.data.preprocess import load_processed
data = load_processed("data/processed/processed_mc_rtt.npz")
```

The SNN branch can develop before this is ready by using
`src.utils.mock_data.make_mock_processed_data()`, which produces an
object with identical schema.

## Citation

```
@inproceedings{nlb2021,
  author    = {Pei, Felix and others},
  title     = {Neural Latents Benchmark: Evaluating Latent Variable Models of Neural Population Activity},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021}
}
```

