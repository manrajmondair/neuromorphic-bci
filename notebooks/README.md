# Notebooks

Per-stage exploration notebooks. Owned by the branch that owns the
corresponding `src/` module.

| Notebook | Owner |
|---|---|
| `01_data_exploration.ipynb` | data-ridge-baseline |
| `02_ridge_baseline.ipynb` | data-ridge-baseline |
| `03_latency_features.ipynb` | snn-latency-decoder |
| `04_snn_decoder.ipynb` | snn-latency-decoder |

Notebooks should call into `src/`. Don't reimplement decoding logic inline.
