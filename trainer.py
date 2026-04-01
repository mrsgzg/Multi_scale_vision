import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

CUR_DIR = os.path.dirname(__file__)
import sys

sys.path.append(CUR_DIR)
sys.path.append(os.path.join(CUR_DIR, "Models"))
sys.path.append(os.path.join(CUR_DIR, "Data_loader"))

from Data_loader_embodiment import get_ball_counting_data_loaders  # type: ignore
from Model_alexnet_embodiment import create_model as create_stepwise_model  # type: ignore
from baselines import create_baseline_model  # type: ignore


@dataclass
class EpochStats:
    loss: float
    final_accuracy: float
    macro_f1: float
    balanced_accuracy: float
    step_accuracy: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PaperTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        config: Dict[str, Any],
        save_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.save_dir = save_dir
        self.ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
            betas=tuple(config.get("adam_betas", [0.9, 0.999])),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("total_epochs", 50),
        )
        self.grad_clip_norm = config.get("grad_clip_norm", 1.0)
        self.best_val_loss = float("inf")

    def _extract_targets(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Main paper target is sample-level final count label.
        return batch["label"].to(self.device).long()

    def _compute_batch_outputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        sequence_data = batch["sequence_data"]
        sequence_data = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in sequence_data.items()
        }
        return self.model(sequence_data)

    def _step_accuracy_vector(
        self,
        sequence_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # sequence_logits: [B, S, C], labels: [B]
        preds = sequence_logits.argmax(dim=-1)
        expanded_labels = labels.unsqueeze(1).expand_as(preds)
        return (preds == expanded_labels).float().mean(dim=0)

    def _run_epoch(self, epoch: int, train: bool) -> Tuple[EpochStats, Dict[str, np.ndarray]]:
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        mode = "Train" if train else "Val"

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_step_acc = []

        pbar = tqdm(loader, desc=f"{mode} {epoch}")
        for batch in pbar:
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                outputs = self._compute_batch_outputs(batch)
                labels = self._extract_targets(batch)

                logits = outputs["logits"]
                loss = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    if self.grad_clip_norm and self.grad_clip_norm > 0:
                        clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            if "sequence_logits" in outputs:
                step_acc = self._step_accuracy_vector(outputs["sequence_logits"], labels)
                all_step_acc.append(step_acc.detach().cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if train:
            self.scheduler.step()

        all_preds_np = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)

        final_accuracy = float((all_preds_np == all_labels_np).mean())
        macro_f1 = float(f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0))
        balanced_acc = float(balanced_accuracy_score(all_labels_np, all_preds_np))

        if all_step_acc:
            step_accuracy = np.stack(all_step_acc).mean(axis=0).tolist()
        else:
            step_accuracy = []

        avg_loss = total_loss / max(1, len(loader))
        stats = EpochStats(
            loss=avg_loss,
            final_accuracy=final_accuracy,
            macro_f1=macro_f1,
            balanced_accuracy=balanced_acc,
            step_accuracy=step_accuracy,
        )
        pred_pack = {
            "preds": all_preds_np,
            "labels": all_labels_np,
        }
        return stats, pred_pack

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        filename = "best.pt" if is_best else f"epoch_{epoch + 1}.pt"
        torch.save(ckpt, os.path.join(self.ckpt_dir, filename))

    def fit(self) -> Dict[str, Any]:
        epochs = self.config.get("total_epochs", 50)
        save_every = self.config.get("save_every", 10)

        history = []
        best_pack = None

        for epoch in range(epochs):
            train_stats, _ = self._run_epoch(epoch, train=True)
            val_stats, val_pack = self._run_epoch(epoch, train=False)

            row = {
                "epoch": epoch,
                "train_loss": train_stats.loss,
                "train_final_accuracy": train_stats.final_accuracy,
                "train_macro_f1": train_stats.macro_f1,
                "train_balanced_accuracy": train_stats.balanced_accuracy,
                "val_loss": val_stats.loss,
                "val_final_accuracy": val_stats.final_accuracy,
                "val_macro_f1": val_stats.macro_f1,
                "val_balanced_accuracy": val_stats.balanced_accuracy,
                "val_step_accuracy": val_stats.step_accuracy,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(row)

            is_best = val_stats.loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_stats.loss
                best_pack = {
                    "preds": val_pack["preds"],
                    "labels": val_pack["labels"],
                    "step_accuracy": val_stats.step_accuracy,
                }
                self._save_checkpoint(epoch, is_best=True)

            if save_every and (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

        out_history = os.path.join(self.save_dir, "history.json")
        with open(out_history, "w") as f:
            json.dump(history, f, indent=2)

        if best_pack is not None:
            np.savez(
                os.path.join(self.save_dir, "best_val_predictions.npz"),
                preds=best_pack["preds"],
                labels=best_pack["labels"],
                step_accuracy=np.array(best_pack["step_accuracy"], dtype=np.float32),
            )

        final_metrics = history[-1]
        with open(os.path.join(self.save_dir, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)

        return {
            "history": history,
            "final_metrics": final_metrics,
            "best_val_loss": self.best_val_loss,
        }


def build_data_loaders(cfg: Dict[str, Any]):
    return get_ball_counting_data_loaders(
        train_csv_path=cfg["train_csv"],
        val_csv_path=cfg["val_csv"],
        data_root=cfg["data_root"],
        batch_size=cfg.get("batch_size", 16),
        sequence_length=cfg.get("sequence_length", 11),
        num_workers=cfg.get("num_workers", 2),
        normalize_images=cfg.get("normalize_images", True),
        custom_image_norm_stats=cfg.get("custom_image_norm_stats", None),
    )


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    model_name = cfg["model_name"]
    model_config = cfg["model_config"].copy()

    if model_name == "dual_stream_stepwise_fusion":
        wrapper_cfg = {
            "image_mode": "rgb",
            "model_config": model_config,
        }
        model = create_stepwise_model(wrapper_cfg, model_type="baseline")
    else:
        model = create_baseline_model(model_name=model_name, model_config=model_config)

    return model.to(device)


def run_training_once(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_root = config.get("experiments_root", os.path.join(CUR_DIR, "experiments"))
    exp_name = config["experiment_name"]
    run_name = f"seed_{seed}"
    save_dir = os.path.join(exp_root, exp_name, run_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train_loader, val_loader = build_data_loaders(config)
    model = build_model(config, device)

    trainer = PaperTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        save_dir=save_dir,
    )

    t0 = time.time()
    out = trainer.fit()
    elapsed = time.time() - t0

    summary = {
        "experiment_name": exp_name,
        "model_name": config["model_name"],
        "seed": seed,
        "save_dir": save_dir,
        "elapsed_seconds": elapsed,
        "best_val_loss": out["best_val_loss"],
        "final_metrics": out["final_metrics"],
    }
    with open(os.path.join(save_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 连续跑多个实验时释放显存，降低 OOM 风险。
    del trainer
    del model
    del train_loader
    del val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one experiment config.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment JSON config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed

    summary = run_training_once(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
