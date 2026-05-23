"""Train and evaluate the sparse spike-latency SNN and the order-shuffle control.

Owned by snn-latency-decoder.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controls.order_shuffle import shuffle_within_bin_order
from src.data.preprocess import load_processed
from src.evaluation.experiment_runner import append_result
from src.evaluation.metrics import velocity_r2
from src.features.event_budget import apply_event_budget
from src.models.snn_decoder import SparseLatencySNN
from src.utils.seed import set_global_seed

EVENT_BUDGETS = [1.00, 0.50, 0.25, 0.10]
SEEDS = [0, 1, 2]


def main() -> None:
    data = load_processed(Path("data/processed/processed_mc_rtt.npz"))
    num_neurons = data["num_neurons"]
    results_csv = Path("results/snn/results.csv")

    for seed in SEEDS:
        set_global_seed(seed)
        for f in EVENT_BUDGETS:
            et, en = apply_event_budget(data["event_times"], data["event_neurons"], f)

            snn = SparseLatencySNN(num_neurons=num_neurons, seed=seed).fit(
                et, en, data["velocity"], data["train_idx"], data["val_idx"]
            )
            y_pred = snn.predict(et, en, data["test_idx"])
            r2 = velocity_r2(data["velocity"][data["test_idx"]], y_pred)
            append_result(
                results_csv,
                {
                    "model": "snn",
                    "event_budget": f,
                    "seed": seed,
                    **r2,
                    "n_events_used": int(sum(t.size for t in et)),
                    "n_events_total": int(sum(t.size for t in data["event_times"])),
                    "notes": "",
                },
            )
            print(f"snn f={f:.2f} seed={seed} r2_joint={r2['r2_joint']:.4f}")

            et_s, en_s = shuffle_within_bin_order(et, en, seed=seed)
            snn_s = SparseLatencySNN(num_neurons=num_neurons, seed=seed).fit(
                et_s, en_s, data["velocity"], data["train_idx"], data["val_idx"]
            )
            y_pred_s = snn_s.predict(et_s, en_s, data["test_idx"])
            r2_s = velocity_r2(data["velocity"][data["test_idx"]], y_pred_s)
            append_result(
                results_csv,
                {
                    "model": "snn_shuffle",
                    "event_budget": f,
                    "seed": seed,
                    **r2_s,
                    "n_events_used": int(sum(t.size for t in et_s)),
                    "n_events_total": int(sum(t.size for t in data["event_times"])),
                    "notes": "within-bin order permuted",
                },
            )
            print(f"snn_shuffle f={f:.2f} seed={seed} r2_joint={r2_s['r2_joint']:.4f}")


if __name__ == "__main__":
    main()
