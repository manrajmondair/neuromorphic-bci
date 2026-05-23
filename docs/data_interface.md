# Shared processed-data interface

This is the single contract every model in the project must respect. If you
change anything here, both sides break — coordinate first.

## The object

`scripts/preprocess_mc_rtt.py` writes one file:

```
data/processed/processed_mc_rtt.npz
```

When loaded with `np.load(..., allow_pickle=True)`, it has these keys:

| Key | Type | Shape / spec |
|---|---|---|
| `spike_counts` | `np.ndarray`, int32 | `[num_bins, num_neurons]` — binned spike counts at `bin_size_ms` |
| `event_times` | `np.ndarray`, object | length `num_bins`; entry `t` is `np.ndarray[float32]` of spike times **within bin t**, in milliseconds, **sorted ascending** |
| `event_neurons` | `np.ndarray`, object | length `num_bins`; entry `t` is `np.ndarray[int32]` of neuron ids, **aligned 1-to-1 with `event_times[t]`** |
| `velocity` | `np.ndarray`, float32 | `[num_bins, 2]` — columns `[vx, vy]`, the decoding target |
| `train_idx` | `np.ndarray`, int64 | bin indices for training |
| `val_idx` | `np.ndarray`, int64 | bin indices for validation |
| `test_idx` | `np.ndarray`, int64 | bin indices for held-out test |
| `bin_size_ms` | `np.ndarray`, scalar | `50` for the headline run |
| `num_neurons` | `np.ndarray`, scalar | int |
| `dataset_name` | `np.ndarray`, scalar str | `"NLB_MC_RTT"` |

## Invariants both sides depend on

1. `event_times[t][k]` and `event_neurons[t][k]` describe the **same** spike.
2. `event_times[t]` is sorted ascending — so event-budget filtering is
   `event_times[t][:k]` where `k = max(1, int(f * len(event_times[t])))`.
3. `spike_counts[t, n] == sum(event_neurons[t] == n)` — the two
   representations must agree, or one side will silently disagree with the
   other under event-budget sweeps.
4. `train_idx`, `val_idx`, `test_idx` are disjoint and cover bins used in
   evaluation. All models train/evaluate on these same splits.
5. Bin `t` has spike events in `[t * bin_size_ms, (t+1) * bin_size_ms)`. The
   times stored in `event_times[t]` are **relative to the start of bin t**,
   so all values are in `[0, bin_size_ms)`.

## Results schema

Every experiment writes one row per (model, event_budget, seed) to a CSV in
`results/{model}/results.csv`:

```
model,event_budget,seed,r2_vx,r2_vy,r2_joint,n_events_used,n_events_total,notes
ridge,1.00,0,0.412,0.388,0.401,123456,123456,
ridge,0.50,0,0.361,0.342,0.352,61728,123456,
snn,1.00,0,0.378,0.351,0.365,123456,123456,
snn_shuffle,0.50,0,0.198,0.182,0.190,61728,123456,within-bin order permuted
```

`r2_joint` is the joint formula from proposal §4.1:
`R² = 1 - Σ_t ||v_t - v̂_t||² / Σ_t ||v_t - v̄||²` (sum runs over both axes).
`r2_vx` and `r2_vy` are the per-axis decompositions, kept for
interpretability. `n_events_used` and `n_events_total` are secondary
efficiency reporting. Use exactly these column names.

In addition to the CSV, each model writes a canonical JSON tracking file
at `results/{model}/{model}_results.json`:

```json
{
  "model": "ridge",
  "metric": "velocity_r2",
  "dataset": "NLB_MC_RTT",
  "config": { "bin_size_ms": 50, "event_budgets": [...], "seeds": [...], ... },
  "results": [
    {"model": "ridge", "event_budget": 1.00, "seed": 0,
     "r2_vx": 0.412, "r2_vy": 0.388, "r2_joint": 0.401,
     "best_alpha": 1.0, "n_events_used": 123456, "n_events_total": 123456,
     "notes": ""}
  ]
}
```

The JSON is what `scripts/make_figures.py` reads to draw the headline
accuracy-efficiency frontier. Use this exact schema for both ridge and
SNN runs so curves overlay automatically.

## Velocity computation

Cursor velocity at bin `t` is computed from cursor position by centered
finite difference, then resampled onto bin centers:

```
v_t = (pos_{t+1} - pos_{t-1}) / (2 * bin_size_s)
```

Edge bins use forward/backward difference. Velocity is **not** smoothed
before being used as the target.

## Train/val/test split

Time-contiguous, not random — random would leak across nearby bins. The
default is the first 70% of bins → train, next 15% → val, last 15% → test,
on the chronological order of the recording. Reproduce by seeding from
`configs/default.yaml`.

## Mock data

While the real preprocessor is being built, the SNN side should develop
against `src.utils.mock_data.make_mock_processed_data()`, which returns an
object with identical keys, shapes, and invariants. When real data is
ready, the SNN code should plug in with a single import swap.
