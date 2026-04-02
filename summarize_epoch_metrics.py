import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Tuple


def _safe_std(values: List[float]) -> float:
    return 0.0 if len(values) <= 1 else pstdev(values)


def _collect_histories(experiments_root: str) -> List[Tuple[str, str, str]]:
    pattern = os.path.join(experiments_root, "*", "seed_*", "history.json")
    paths = sorted(glob.glob(pattern))
    out = []
    for p in paths:
        run_dir = os.path.dirname(p)
        seed_name = os.path.basename(run_dir)
        exp_name = os.path.basename(os.path.dirname(run_dir))
        out.append((exp_name, seed_name, p))
    return out


def _epoch_to_index(epoch_value: int, one_based: bool) -> int:
    return epoch_value - 1 if one_based else epoch_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model performance at selected epochs from existing history.json files")
    parser.add_argument("--experiments_root", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--one_based", action="store_true", help="Treat epoch values as 1-based (recommended for paper reporting)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    histories = _collect_histories(args.experiments_root)
    if not histories:
        raise RuntimeError(f"No history.json found under: {args.experiments_root}")

    raw_rows: List[Dict] = []

    for exp_name, seed_name, history_path in histories:
        with open(history_path, "r") as f:
            history = json.load(f)

        for epoch_value in args.epochs:
            idx = _epoch_to_index(epoch_value, one_based=args.one_based)
            if idx < 0 or idx >= len(history):
                raw_rows.append(
                    {
                        "experiment_name": exp_name,
                        "seed": seed_name,
                        "epoch": epoch_value,
                        "available": 0,
                        "val_loss": "",
                        "val_final_accuracy": "",
                        "val_macro_f1": "",
                        "val_balanced_accuracy": "",
                    }
                )
                continue

            row = history[idx]
            raw_rows.append(
                {
                    "experiment_name": exp_name,
                    "seed": seed_name,
                    "epoch": epoch_value,
                    "available": 1,
                    "val_loss": row.get("val_loss", ""),
                    "val_final_accuracy": row.get("val_final_accuracy", ""),
                    "val_macro_f1": row.get("val_macro_f1", ""),
                    "val_balanced_accuracy": row.get("val_balanced_accuracy", ""),
                }
            )

    raw_csv = os.path.join(args.results_dir, "epoch_metrics_raw.csv")
    raw_fields = [
        "experiment_name",
        "seed",
        "epoch",
        "available",
        "val_loss",
        "val_final_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
    ]
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped = defaultdict(list)
    for r in raw_rows:
        key = (r["experiment_name"], r["epoch"])
        grouped[key].append(r)

    summary_rows = []
    for (exp_name, epoch_value), rows in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        valid = [r for r in rows if r["available"] == 1]
        if not valid:
            summary_rows.append(
                {
                    "experiment_name": exp_name,
                    "epoch": epoch_value,
                    "n_available_seeds": 0,
                    "val_loss_mean": "",
                    "val_loss_std": "",
                    "val_final_accuracy_mean": "",
                    "val_final_accuracy_std": "",
                    "val_macro_f1_mean": "",
                    "val_macro_f1_std": "",
                    "val_balanced_accuracy_mean": "",
                    "val_balanced_accuracy_std": "",
                }
            )
            continue

        val_loss = [float(r["val_loss"]) for r in valid]
        val_acc = [float(r["val_final_accuracy"]) for r in valid]
        val_f1 = [float(r["val_macro_f1"]) for r in valid]
        val_bal = [float(r["val_balanced_accuracy"]) for r in valid]

        summary_rows.append(
            {
                "experiment_name": exp_name,
                "epoch": epoch_value,
                "n_available_seeds": len(valid),
                "val_loss_mean": mean(val_loss),
                "val_loss_std": _safe_std(val_loss),
                "val_final_accuracy_mean": mean(val_acc),
                "val_final_accuracy_std": _safe_std(val_acc),
                "val_macro_f1_mean": mean(val_f1),
                "val_macro_f1_std": _safe_std(val_f1),
                "val_balanced_accuracy_mean": mean(val_bal),
                "val_balanced_accuracy_std": _safe_std(val_bal),
            }
        )

    summary_csv = os.path.join(args.results_dir, "epoch_metrics_summary.csv")
    summary_fields = [
        "experiment_name",
        "epoch",
        "n_available_seeds",
        "val_loss_mean",
        "val_loss_std",
        "val_final_accuracy_mean",
        "val_final_accuracy_std",
        "val_macro_f1_mean",
        "val_macro_f1_std",
        "val_balanced_accuracy_mean",
        "val_balanced_accuracy_std",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Wrote:")
    print(f"- {raw_csv}")
    print(f"- {summary_csv}")


if __name__ == "__main__":
    main()
