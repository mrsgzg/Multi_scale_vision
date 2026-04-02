import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from Data_loader.Data_loader_embodiment import get_ball_counting_data_loaders  # type: ignore

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
from trainer import build_model  # type: ignore


DEFAULT_EXPERIMENTS = [
    "joint_only",
    "visual_only",
    "early_fusion_single_stream",
    "dual_stream_late_fusion",
    "dual_stream_stepwise_fusion",
]


def load_run_config(run_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"run_config.json not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def resolve_run_dir(experiments_root: str, experiment_name: str, seed: int) -> str:
    run_dir = os.path.join(experiments_root, experiment_name, f"seed_{seed}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    return run_dir


def resolve_checkpoint_path(run_dir: str, checkpoint_name: str) -> str:
    ckpt_path = os.path.join(run_dir, "checkpoints", checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    return ckpt_path


def build_checkpoint_list(checkpoint_name: str, epochs: List[int]) -> List[str]:
    if epochs:
        return [f"epoch_{int(e)}.pt" for e in epochs]
    return [checkpoint_name]


def checkpoint_tag(checkpoint_name: str) -> str:
    m = re.match(r"^epoch_(\d+)\.pt$", checkpoint_name)
    if m:
        return f"epoch_{m.group(1)}"
    return os.path.splitext(os.path.basename(checkpoint_name))[0]


def build_val_loader_from_config(cfg: Dict[str, Any], eval_batch_size: int):
    _, val_loader = get_ball_counting_data_loaders(
        train_csv_path=cfg["train_csv"],
        val_csv_path=cfg["val_csv"],
        data_root=cfg["data_root"],
        batch_size=eval_batch_size,
        sequence_length=cfg.get("sequence_length", 11),
        num_workers=cfg.get("num_workers", 2),
        normalize_images=cfg.get("normalize_images", True),
        custom_image_norm_stats=cfg.get("custom_image_norm_stats", None),
    )
    return val_loader


def is_non_decreasing(arr: np.ndarray) -> bool:
    return bool(np.all(arr[1:] >= arr[:-1]))


def collect_sample_trajectories(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in val_loader:
            sequence_data = batch["sequence_data"]
            sequence_data = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in sequence_data.items()
            }

            outputs = model(sequence_data)
            if "sequence_logits" not in outputs:
                raise RuntimeError("Model output missing sequence_logits")

            sequence_logits = outputs["sequence_logits"]
            sequence_probs = F.softmax(sequence_logits, dim=-1)
            sequence_preds = sequence_logits.argmax(dim=-1)

            labels = batch["label"].numpy()
            sample_ids = batch["sample_id"]

            batch_size = sequence_preds.shape[0]
            seq_len = sequence_preds.shape[1]

            for i in range(batch_size):
                pred_curve = sequence_preds[i].detach().cpu().numpy().astype(int)
                true_count = int(labels[i])
                sid = str(sample_ids[i])

                true_prob_curve = sequence_probs[i, :, true_count].detach().cpu().numpy()
                final_pred = int(pred_curve[-1])

                rec: Dict[str, Any] = {
                    "sample_id": sid,
                    "true_count": true_count,
                    "final_pred": final_pred,
                    "final_match": int(final_pred == true_count),
                    "non_decreasing": int(is_non_decreasing(pred_curve)),
                }

                for t in range(seq_len):
                    rec[f"step_{t+1}_pred"] = int(pred_curve[t])
                    rec[f"step_{t+1}_true_prob"] = float(true_prob_curve[t])

                records.append(rec)

    return records


def select_records_for_plot(records: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    """优先按不同 true_count 选样本，再补齐到 max_samples。"""
    if not records:
        return []

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(int(rec["true_count"]), []).append(rec)

    selected: List[Dict[str, Any]] = []
    # 先每个类别取一个，保证 label 多样性
    for label in sorted(grouped.keys()):
        selected.append(grouped[label][0])
        if len(selected) >= max_samples:
            return selected

    # 再按原顺序补齐
    for rec in records:
        if rec in selected:
            continue
        selected.append(rec)
        if len(selected) >= max_samples:
            break

    return selected


def render_sample_plot(records: List[Dict[str, Any]], out_png: str) -> None:
    if not records:
        raise RuntimeError("No records to plot")

    n = len(records)
    step_cols = sorted([k for k in records[0].keys() if k.startswith("step_") and k.endswith("_pred")], key=lambda x: int(x.split("_")[1]))
    seq_len = len(step_cols)
    steps = np.arange(1, seq_len + 1)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, max(8, 2.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for i, rec in enumerate(records):
        ax = axes[i]
        preds = np.array([rec[c] for c in step_cols], dtype=int)
        true_count = rec["true_count"]

        ax.plot(steps, preds, marker="o", linewidth=1.8, color="#4C72B0")
        ax.axhline(y=true_count, linestyle="--", linewidth=1.2, color="#C44E52")
        ax.set_ylabel(f"S{i+1}")
        ax.set_ylim(-0.5, 10.5)
        ax.grid(alpha=0.2)
        ax.set_title(
            f"sample_id={rec['sample_id']} | true={true_count} | final={rec['final_pred']} | non_dec={rec['non_decreasing']}",
            fontsize=9,
        )

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Step-by-step predicted count trajectories", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def collect_all_labels_preds(model: torch.nn.Module, val_loader, device: torch.device):
    """Run model on full validation set, return (labels, preds) arrays using final-step prediction."""
    model.eval()
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            sequence_data = batch["sequence_data"]
            sequence_data = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in sequence_data.items()
            }
            outputs = model(sequence_data)
            sequence_logits = outputs["sequence_logits"]
            final_preds = sequence_logits[:, -1, :].argmax(dim=-1).cpu().numpy()
            all_preds.append(final_preds)
            all_labels.append(batch["label"].numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def plot_epoch_confusion_matrices(
    experiments_root: str,
    results_dir: str,
    exp_names: List[str],
    seed: int,
    epochs: List[int],
    device: torch.device,
    eval_batch_size: int = 16,
) -> List[str]:
    """Load checkpoint at each epoch for each model and generate confusion matrix PNGs.

    Output files: cm_{exp_name}_epoch_{epoch}.png in results_dir.
    These correspond to the epoch-based confusion matrix figure in the paper.
    """
    out_paths: List[str] = []
    for exp_name in exp_names:
        for epoch in epochs:
            try:
                run_dir = resolve_run_dir(experiments_root, exp_name, seed)
                cfg = load_run_config(run_dir)
                ckpt_name = f"epoch_{epoch}.pt"
                ckpt_path = resolve_checkpoint_path(run_dir, ckpt_name)

                model = build_model(cfg, device)
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])

                val_loader = build_val_loader_from_config(cfg, eval_batch_size=eval_batch_size)
                labels, preds = collect_all_labels_preds(model, val_loader, device)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                classes = list(range(int(max(labels.max(), preds.max())) + 1))
                cm = confusion_matrix(labels, preds, labels=classes)

                title = f"{_DISPLAY_NAMES.get(exp_name, exp_name)} — Epoch {epoch}"
                plt.figure(figsize=(8, 6.5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    annot_kws={"size": 12},
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(title)
                plt.tight_layout()

                out_path = os.path.join(results_dir, f"cm_{exp_name}_epoch_{epoch}.png")
                plt.savefig(out_path, dpi=300)
                plt.close()
                out_paths.append(out_path)
                print(f"  CM saved: {out_path}")
            except Exception as e:
                print(f"  CM FAILED {exp_name} epoch {epoch}: {e}")
    return out_paths



    parser = argparse.ArgumentParser(description="Visualize per-step predictions from existing checkpoint without retraining")
    parser.add_argument("--experiments_root", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--experiment_names", type=str, nargs="+", default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_name", type=str, default="best.pt")
    parser.add_argument("--epochs", type=int, nargs="+", default=None, help="If provided, run visualization for each epoch checkpoint, e.g. --epochs 20 40 80")
    parser.add_argument("--max_samples", type=int, default=10, help="Number of samples shown in PNG; CSV always contains full validation set")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.all_models:
        experiment_names = DEFAULT_EXPERIMENTS
    elif args.experiment_names:
        experiment_names = args.experiment_names
    elif args.experiment_name:
        experiment_names = [args.experiment_name]
    else:
        experiment_names = ["dual_stream_stepwise_fusion"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    written_files = []
    failed = []

    for exp_name in experiment_names:
        try:
            run_dir = resolve_run_dir(args.experiments_root, exp_name, args.seed)
            cfg = load_run_config(run_dir)
            checkpoints = build_checkpoint_list(args.checkpoint_name, args.epochs or [])

            for ckpt_name in checkpoints:
                model = build_model(cfg, device)
                ckpt_path = resolve_checkpoint_path(run_dir, ckpt_name)
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])

                val_loader = build_val_loader_from_config(cfg, eval_batch_size=args.eval_batch_size)
                records = collect_sample_trajectories(model, val_loader, device)

                tag = checkpoint_tag(ckpt_name)
                out_csv = os.path.join(
                    args.results_dir,
                    f"checkpoint_sample_steps_{exp_name}_seed_{args.seed}_{tag}.csv",
                )
                pd.DataFrame(records).to_csv(out_csv, index=False)

                out_png = os.path.join(
                    args.results_dir,
                    f"checkpoint_sample_steps_{exp_name}_seed_{args.seed}_{tag}.png",
                )
                plot_records = select_records_for_plot(records, args.max_samples)
                render_sample_plot(plot_records, out_png)

                written_files.extend([out_csv, out_png])
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            failed.append((exp_name, str(e)))

    print("Wrote:")
    for p in written_files:
        print(f"- {p}")

    if failed:
        print("Failed:")
        for name, msg in failed:
            print(f"- {name}: {msg}")


if __name__ == "__main__":
    main()
