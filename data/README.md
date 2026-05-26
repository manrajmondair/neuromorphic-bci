# Data

This directory holds the dataset locally. **Nothing in `raw/` or
`processed/` is committed** — both are gitignored. Each contributor
downloads their own copy.

## What dataset / how to get it

See **`docs/dataset.md`** — that's the single source of truth for which
dataset we use, where it lives, and how to download it. Don't duplicate
dataset info here; this README only describes the local layout.

## Layout

```
data/
├── raw/           dandi download lands here  (gitignored)
│   └── 000129/    NLB MC_RTT dandiset
└── processed/     preprocess script output    (gitignored)
    └── processed_mc_rtt.npz   schema in docs/data_interface.md
```

## One-line bootstrap

```bash
python scripts/download_mc_rtt.py && python scripts/preprocess_mc_rtt.py
```

After that, both ridge and SNN training scripts work against
`data/processed/processed_mc_rtt.npz`.
