import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_main_bars(main_table_csv: str, results_dir: str):
    df = pd.read_csv(main_table_csv)

    order = [
        "joint_only",
        "visual_only",
        "early_fusion_single_stream",
        "dual_stream_late_fusion",
        "dual_stream_stepwise_fusion",
    ]
    df = df.set_index("experiment_name").reindex(order).dropna().reset_index()

    plt.figure(figsize=(10, 5))
    plt.bar(
        df["experiment_name"],
        df["final_accuracy_mean"],
        yerr=df["final_accuracy_std"],
        capsize=4,
        color="#4C72B0",
    )
    plt.ylabel("Final Accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out1 = os.path.join(results_dir, "final_accuracy_bar.png")
    plt.savefig(out1, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(
        df["experiment_name"],
        df["macro_f1_mean"],
        yerr=df["macro_f1_std"],
        capsize=4,
        color="#55A868",
    )
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out2 = os.path.join(results_dir, "macro_f1_bar.png")
    plt.savefig(out2, dpi=300)
    plt.close()

    return out1, out2


def plot_accuracy_over_time(experiments_root: str, results_dir: str):
    model_map = {
        "early_fusion": "early_fusion_single_stream",
        "late_fusion": "dual_stream_late_fusion",
        "stepwise_fusion": "dual_stream_stepwise_fusion",
    }

    curves = {}
    for label, exp_name in model_map.items():
        exp_dir = os.path.join(experiments_root, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        step_curves = []
        for run_name in os.listdir(exp_dir):
            pred_path = os.path.join(exp_dir, run_name, "best_val_predictions.npz")
            if os.path.exists(pred_path):
                pack = np.load(pred_path)
                if "step_accuracy" in pack and len(pack["step_accuracy"]) > 0:
                    step_curves.append(pack["step_accuracy"])

        if step_curves:
            curves[label] = np.mean(np.stack(step_curves), axis=0)

    if not curves:
        return None

    plt.figure(figsize=(8, 5))
    for label, curve in curves.items():
        steps = np.arange(1, len(curve) + 1)
        plt.plot(steps, curve, marker="o", label=label)

    plt.xlabel("Time step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(results_dir, "accuracy_over_time.png")
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_confusion_matrix_for_exp(experiments_root: str, experiment_name: str, out_path: str):
    exp_dir = os.path.join(experiments_root, experiment_name)
    if not os.path.isdir(exp_dir):
        return None

    all_preds = []
    all_labels = []
    for run_name in os.listdir(exp_dir):
        pred_path = os.path.join(exp_dir, run_name, "best_val_predictions.npz")
        if os.path.exists(pred_path):
            pack = np.load(pred_path)
            all_preds.append(pack["preds"])
            all_labels.append(pack["labels"])

    if not all_preds:
        return None

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    classes = list(range(int(max(labels.max(), preds.max())) + 1))
    cm = confusion_matrix(labels, preds, labels=classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(experiment_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Create paper-ready visualizations from aggregated runs")
    parser.add_argument("--experiments_root", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    main_table_csv = os.path.join(args.results_dir, "main_table.csv")
    if not os.path.exists(main_table_csv):
        raise RuntimeError("main_table.csv not found. Run evaluate.py first.")

    out_files = []
    out_files.extend(plot_main_bars(main_table_csv, args.results_dir))

    over_time = plot_accuracy_over_time(args.experiments_root, args.results_dir)
    if over_time:
        out_files.append(over_time)

    cm_visual = plot_confusion_matrix_for_exp(
        args.experiments_root,
        "visual_only",
        os.path.join(args.results_dir, "cm_visual_only.png"),
    )
    if cm_visual:
        out_files.append(cm_visual)

    cm_stepwise = plot_confusion_matrix_for_exp(
        args.experiments_root,
        "dual_stream_stepwise_fusion",
        os.path.join(args.results_dir, "cm_stepwise_fusion.png"),
    )
    if cm_stepwise:
        out_files.append(cm_stepwise)

    with open(os.path.join(args.results_dir, "figures_manifest.json"), "w") as f:
        json.dump({"files": out_files}, f, indent=2)

    print("Wrote figures:")
    for path in out_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
