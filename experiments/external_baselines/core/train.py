import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast

from core.dataset import create_dataloaders, run_offline_preprocessing
from core.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience: int, path: str):
        self.patience = int(patience)
        self.path = path
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best is None or score > self.best:
            self.best = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True


def build_criterion(cfg: dict, device: torch.device):
    weights = cfg["train"].get("class_weights", None)
    if weights:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


def run_epoch(model, loaders, criterion, device, optimizer=None, scaler=None, amp_enabled=False):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for loader in loaders.values():
        for batch in loader:
            if batch is None:
                continue
            img, sig, lbl = batch[:3]
            img = img.to(device, non_blocking=True)
            sig = sig.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                logits = model(img, sig)
                loss = criterion(logits, lbl)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            preds = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * lbl.size(0)
            total_correct += int((preds == lbl).sum().item())
            total_count += int(lbl.size(0))

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def evaluate(model, loaders, device, class_names, out_dir):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for loader in loaders.values():
            for batch in loader:
                if batch is None:
                    continue
                img, sig, lbl = batch[:3]
                logits = model(img.to(device, non_blocking=True), sig.to(device, non_blocking=True))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(lbl.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "classification_report_clean.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (clean)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_clean.png"), dpi=150)
    plt.close(fig)

    return {
        "accuracy_pct": float(acc * 100.0),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_test_samples": int(len(y_true)),
    }


def save_run_row(cfg: dict, metrics: dict, out_dir: str):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    row = {
        "experiment": str(cfg.get("project", {}).get("exp_id", cfg.get("project", {}).get("version", "unknown"))),
        "model": str(cfg.get("model", {}).get("name", "unknown")),
        "split_mode": str(data_cfg.get("split_mode", "temporal")),
        "seed": int(train_cfg.get("seed", 42)),
        "signal_preprocess": str(data_cfg.get("signal_preprocess", "linear")),
        "image_preprocess": str(data_cfg.get("image_preprocess", "none")),
        "accuracy_pct": metrics["accuracy_pct"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "n_test_samples": metrics["n_test_samples"],
        "snr_db": "clean",
    }

    csv_path = os.path.join(out_dir, "all_runs.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    with open(os.path.join(out_dir, "metrics_clean.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=".")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    cfg_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if bool(cfg["data"].get("use_offline", False)):
        run_offline_preprocessing(cfg)

    device = torch.device(cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    log_dir = cfg["train"].get("log_dir", "./logs")
    model_dir = cfg["train"].get("model_dir", "./checkpoints")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(exp_dir, log_dir)
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(exp_dir, model_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_loaders, val_loaders, test_loaders = create_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = build_criterion(cfg, device)
    opt_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"].get("learning_rate", 2e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if bool(cfg["train"].get("use_cosine", True)):
        epochs = int(cfg["train"].get("epochs", 100))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    scaler = GradScaler(enabled=amp_enabled)
    best_path = os.path.join(model_dir, cfg["train"].get("best_model_name", "best_model.pth"))
    stopper = EarlyStopping(patience=int(cfg["train"].get("early_stopping_patience", 20)), path=best_path)

    epochs = int(cfg["train"].get("epochs", 100))
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loaders, criterion, device, optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled)
        val_loss, val_acc = run_epoch(model, val_loaders, criterion, device, optimizer=None, scaler=None, amp_enabled=amp_enabled)
        if scheduler is not None:
            scheduler.step()
        stopper(val_loss, model)
        logging.info(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | time={time.time()-t0:.1f}s"
        )
        if stopper.stop:
            logging.info("Early stopping triggered")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    class_names = cfg["data"].get("case_ids", [str(i) for i in range(int(cfg["data"].get("num_classes", 6)))])
    metrics = evaluate(model, test_loaders, device, class_names, log_dir)
    save_run_row(cfg, metrics, out_dir=log_dir)
    logging.info(f"Final clean metrics: {metrics}")


if __name__ == "__main__":
    main()
