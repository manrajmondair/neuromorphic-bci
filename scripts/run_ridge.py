"""Train and evaluate the ridge spike-count baseline across event budgets.

Owned by data-ridge-baseline.
"""
from __future__ import annotations

from pathlib import Path

from src.data.preprocess import load_processed
from src.evaluation.experiment_runner import append_result
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.features.spike_counts import counts_from_events
from src.models.ridge_decoder import RidgeDecoder
from src.utils.seed import set_global_seed

EVENT_BUDGETS = [1.00, 0.50, 0.25, 0.10]
SEEDS = [0, 1, 2]


def main() -> None:
    data = load_processed(Path("data/processed/processed_mc_rtt.npz"))
    num_neurons = data["num_neurons"]
    results_csv = Path("results/ridge/results.csv")

    for seed in SEEDS:
        set_global_seed(seed)
        for f in EVENT_BUDGETS:
            et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)
            X = counts_from_events(et, en, num_neurons)
            y = data["velocity"]

            decoder = RidgeDecoder().fit(X[data["train_idx"]], y[data["train_idx"]])
            y_pred = decoder.predict(X[data["test_idx"]])
            r2 = velocity_r2(y[data["test_idx"]], y_pred)

            append_result(
                results_csv,
                {
                    "model": "ridge",
                    "event_budget": f,
                    "seed": seed,
                    **r2,
                    "n_events_used": int(sum(t.size for t in et)),
                    "n_events_total": int(sum(t.size for t in data["event_times"])),
                    "notes": "",
                },
            )
            print(f"ridge f={f:.2f} seed={seed} r2_mean={r2['r2_mean']:.4f}")


if __name__ == "__main__":
    main()
