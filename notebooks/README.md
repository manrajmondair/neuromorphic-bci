# Notebooks

Notebooks call into `src/` — they do not reimplement decoding logic inline.

| Notebook | Purpose | Owner |
|---|---|---|
| `02_ridge_baseline_colab.ipynb` | End-to-end ridge baseline on a Colab GPU with closed-form vectorized alpha sweep over ~1000 regularizers | `data-ridge-baseline` |

Additional per-stage exploration notebooks can be added under either
branch's ownership without coordination — they should never modify the
shared interfaces in `docs/data_interface.md` or `src/evaluation/metrics.py`.
