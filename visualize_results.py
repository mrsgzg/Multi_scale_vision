import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
})

_DISPLAY_NAMES = {
    "joint_only": "E0 Joint-only",
    "visual_only": "E1 Visual-only",
    "early_fusion_single_stream": "E2 Early fusion",
    "dual_stream_late_fusion": "E3 Late fusion",
    "dual_stream_stepwise_fusion": "E4 Stepwise fusion",
}


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

    labels = [_DISPLAY_NAMES.get(n, n) for n in df["experiment_name"]]

    plt.figure(figsize=(10, 5))
    plt.bar(
        labels,
        df["final_accuracy_mean"],
        yerr=df["final_accuracy_std"],
        capsize=5,
        color="#4C72B0",
    )
    plt.ylabel("Final Accuracy")
    plt.xlabel("Model")
    plt.ylim(0.90, 1.02)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out1 = os.path.join(results_dir, "final_accuracy_bar.png")
    plt.savefig(out1, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(
        labels,
        df["macro_f1_mean"],
        yerr=df["macro_f1_std"],
        capsize=5,
        color="#55A868",
    )
    plt.ylabel("Macro-F1")
    plt.xlabel("Model")
    plt.ylim(0.90, 1.02)
    plt.xticks(rotation=15, ha="right")
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

    plt.xlabel("Time Step")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out = os.path.join(results_dir, "accuracy_over_time.png")
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_val_accuracy_learning_curves(experiments_root: str, results_dir: str):
    model_order = [
        "joint_only",
        "visual_only",
        "early_fusion_single_stream",
        "dual_stream_late_fusion",
        "dual_stream_stepwise_fusion",
    ]

    curves = {}
    for exp_name in model_order:
        pattern = os.path.join(experiments_root, exp_name, "seed_*", "history.json")
        history_paths = sorted(glob.glob(pattern))
        if not history_paths:
            continue

        run_curves = []
        for hp in history_paths:
            with open(hp, "r") as f:
                hist = json.load(f)
            vals = [float(row.get("val_final_accuracy", 0.0)) for row in hist]
            if vals:
                run_curves.append(vals)

        if not run_curves:
            continue

        min_len = min(len(x) for x in run_curves)
        if min_len <= 0:
            continue

        arr = np.array([x[:min_len] for x in run_curves], dtype=float)
        curves[exp_name] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "n": arr.shape[0],
        }

    if not curves:
        return None

    def smooth_curve(arr: np.ndarray, window: int = 21) -> np.ndarray:
        if arr.size < 3:
            return arr
        w = min(window, arr.size)
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return arr
        kernel = np.ones(w, dtype=float) / float(w)
        pad = w // 2
        padded = np.pad(arr, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    display_name = {
        "joint_only": "E0 Joint-only",
        "visual_only": "E1 Visual-only",
        "early_fusion_single_stream": "E2 Early fusion",
        "dual_stream_late_fusion": "E3 Late fusion",
        "dual_stream_stepwise_fusion": "E4 Stepwise fusion",
    }

    plt.figure(figsize=(9, 5.5))
    for exp_name in model_order:
        if exp_name not in curves:
            continue
        c = curves[exp_name]
        x = np.arange(1, len(c["mean"]) + 1)
        y = smooth_curve(c["mean"], window=21)
        s = smooth_curve(c["std"], window=21)
        label = display_name.get(exp_name, exp_name)
        plt.plot(x, y, linewidth=2.0, label=label)
        plt.fill_between(x, np.clip(y - s, 0.0, 1.0), np.clip(y + s, 0.0, 1.0), alpha=0.18)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(results_dir, "val_acc_learning_curves.png")
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

    short_title = _DISPLAY_NAMES.get(experiment_name, experiment_name)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(short_title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def _pick_run_for_samples(exp_dir: str):
    run_names = sorted([x for x in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, x))])
    if not run_names:
        return None
    if "seed_42" in run_names:
        return "seed_42"
    return run_names[0]


def plot_sample_step_outputs(experiments_root: str, results_dir: str, experiment_name: str = "dual_stream_stepwise_fusion", max_samples: int = 10):
    exp_dir = os.path.join(experiments_root, experiment_name)
    if not os.path.isdir(exp_dir):
        return []

    run_name = _pick_run_for_samples(exp_dir)
    if run_name is None:
        return []

    pred_path = os.path.join(exp_dir, run_name, "best_val_predictions.npz")
    if not os.path.exists(pred_path):
        return []

    pack = np.load(pred_path, allow_pickle=True)
    if "sequence_preds" not in pack or pack["sequence_preds"].size == 0:
        return []

    sequence_preds = pack["sequence_preds"]
    labels = pack["labels"]
    sample_ids = pack["sample_ids"] if "sample_ids" in pack else np.array([f"idx_{i}" for i in range(len(labels))], dtype=object)

    n = min(max_samples, sequence_preds.shape[0])
    steps = np.arange(1, sequence_preds.shape[1] + 1)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, max(8, 2.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    records = []
    for i in range(n):
        ax = axes[i]
        pred_curve = sequence_preds[i]
        true_count = int(labels[i])
        sid = str(sample_ids[i])

        ax.plot(steps, pred_curve, marker="o", color="#4C72B0", linewidth=1.8)
        ax.axhline(y=true_count, color="#C44E52", linestyle="--", linewidth=1.2)
        ax.set_ylabel(f"S{i+1}")
        ax.set_ylim(-0.5, 10.5)
        ax.grid(alpha=0.2)
        ax.set_title(f"sample_id={sid} | true={true_count}", fontsize=9)

        row = {"sample_index": i + 1, "sample_id": sid, "true_count": true_count}
        for t, pred in enumerate(pred_curve, start=1):
            row[f"step_{t}"] = int(pred)
        records.append(row)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"Step-by-step predicted counts (first {n} samples) - {experiment_name} ({run_name})", fontsize=12)
    plt.tight_layout()

    out_png = os.path.join(results_dir, "sample_steps_1_10.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    out_csv = os.path.join(results_dir, "sample_steps_1_10.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)

    return [out_png, out_csv]


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

    learning_curves = plot_val_accuracy_learning_curves(args.experiments_root, args.results_dir)
    if learning_curves:
        out_files.append(learning_curves)

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

    sample_files = plot_sample_step_outputs(
        args.experiments_root,
        args.results_dir,
        experiment_name="dual_stream_stepwise_fusion",
        max_samples=10,
    )
    out_files.extend(sample_files)

    with open(os.path.join(args.results_dir, "figures_manifest.json"), "w") as f:
        json.dump({"files": out_files}, f, indent=2)

    print("Wrote figures:")
    for path in out_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
