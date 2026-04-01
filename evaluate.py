import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def _safe_std(values: List[float]) -> float:
    return 0.0 if len(values) <= 1 else pstdev(values)


def _collect_run_summaries(experiments_root: str):
    pattern = os.path.join(experiments_root, "*", "seed_*", "run_summary.json")
    run_files = sorted(glob.glob(pattern))
    rows = []

    for path in run_files:
        with open(path, "r") as f:
            summary = json.load(f)

        save_dir = summary["save_dir"]
        pred_path = os.path.join(save_dir, "best_val_predictions.npz")
        if not os.path.exists(pred_path):
            continue

        pred_pack = np.load(pred_path)
        preds = pred_pack["preds"]
        labels = pred_pack["labels"]

        final_acc = float((preds == labels).mean())
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(labels, preds))
        step_accuracy = pred_pack["step_accuracy"].tolist() if "step_accuracy" in pred_pack else []

        rows.append(
            {
                "experiment_name": summary["experiment_name"],
                "model_name": summary["model_name"],
                "seed": summary["seed"],
                "save_dir": save_dir,
                "elapsed_seconds": summary["elapsed_seconds"],
                "best_val_loss": summary["best_val_loss"],
                "final_accuracy": final_acc,
                "macro_f1": macro_f1,
                "balanced_accuracy": bal_acc,
                "step_accuracy": step_accuracy,
            }
        )

    return rows


def _write_metrics_summary(results_dir: str, rows: List[Dict]):
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "metrics_summary.csv")

    fields = [
        "experiment_name",
        "model_name",
        "seed",
        "final_accuracy",
        "macro_f1",
        "balanced_accuracy",
        "best_val_loss",
        "elapsed_seconds",
        "save_dir",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

    return out_csv


def _write_main_table(results_dir: str, rows: List[Dict]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["experiment_name"]].append(row)

    out_csv = os.path.join(results_dir, "main_table.csv")
    fields = [
        "experiment_name",
        "model_name",
        "n_seeds",
        "final_accuracy_mean",
        "final_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "best_val_loss_mean",
        "best_val_loss_std",
        "elapsed_seconds_mean",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for exp_name in sorted(grouped.keys()):
            exp_rows = grouped[exp_name]
            final_accs = [r["final_accuracy"] for r in exp_rows]
            macro_f1s = [r["macro_f1"] for r in exp_rows]
            bal_accs = [r["balanced_accuracy"] for r in exp_rows]
            val_losses = [r["best_val_loss"] for r in exp_rows]
            elapsed = [r["elapsed_seconds"] for r in exp_rows]

            writer.writerow(
                {
                    "experiment_name": exp_name,
                    "model_name": exp_rows[0]["model_name"],
                    "n_seeds": len(exp_rows),
                    "final_accuracy_mean": mean(final_accs),
                    "final_accuracy_std": _safe_std(final_accs),
                    "macro_f1_mean": mean(macro_f1s),
                    "macro_f1_std": _safe_std(macro_f1s),
                    "balanced_accuracy_mean": mean(bal_accs),
                    "balanced_accuracy_std": _safe_std(bal_accs),
                    "best_val_loss_mean": mean(val_losses),
                    "best_val_loss_std": _safe_std(val_losses),
                    "elapsed_seconds_mean": mean(elapsed),
                }
            )

    return out_csv


def _write_run_manifest(results_dir: str, rows: List[Dict]):
    out_json = os.path.join(results_dir, "run_manifest.json")
    manifest = {
        "n_runs": len(rows),
        "runs": rows,
    }
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_json


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics into paper-ready tables")
    parser.add_argument("--experiments_root", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    rows = _collect_run_summaries(args.experiments_root)
    if not rows:
        raise RuntimeError("No run summaries found. Please run training first.")

    summary_csv = _write_metrics_summary(args.results_dir, rows)
    main_table_csv = _write_main_table(args.results_dir, rows)
    manifest_json = _write_run_manifest(args.results_dir, rows)

    print("Wrote:")
    print(f"- {summary_csv}")
    print(f"- {main_table_csv}")
    print(f"- {manifest_json}")


if __name__ == "__main__":
    main()
