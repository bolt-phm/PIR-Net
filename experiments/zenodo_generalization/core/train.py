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
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast

from core.dataset import create_dataloaders, dataset_export_manifest, dataset_summary, run_offline_preprocessing
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
    def __init__(self, patience: int, path: str, mode: str = "max", min_epochs: int = 0):
        self.patience = int(patience)
        self.path = path
        self.mode = mode
        self.min_epochs = int(min_epochs)
        self.best = None
        self.counter = 0
        self.stop = False

    def _is_better(self, score: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return score < self.best
        return score > self.best

    def __call__(self, score: float, model: nn.Module, epoch: int):
        if self._is_better(score):
            self.best = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return
        if int(epoch) < self.min_epochs:
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True


def build_criterion(cfg: dict, device: torch.device):
    weights = cfg["train"].get("class_weights", None)
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    if weights:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def logits_to_preds(logits: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
    if logits.size(1) == 2 and threshold is not None:
        prob_1 = torch.softmax(logits, dim=1)[:, 1]
        return (prob_1 >= threshold).long()
    return torch.argmax(logits, dim=1)


def run_epoch(
    model,
    loaders,
    criterion,
    device,
    n_classes: int,
    optimizer=None,
    scaler=None,
    amp_enabled=False,
    grad_clip_norm: float = 0.0,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    y_true, y_pred = [], []

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
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            preds = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * lbl.size(0)
            y_true.extend(lbl.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    total_count = len(y_true)
    if total_count == 0:
        return 0.0, 0.0, 0.0

    labels = list(range(n_classes))
    avg_loss = total_loss / total_count
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return avg_loss, float(acc), float(macro_f1)


def collect_logits_and_labels(model, loaders, device):
    model.eval()
    all_logits, y_true = [], []
    with torch.no_grad():
        for loader in loaders.values():
            for batch in loader:
                if batch is None:
                    continue
                img, sig, lbl = batch[:3]
                logits = model(img.to(device, non_blocking=True), sig.to(device, non_blocking=True))
                all_logits.append(logits.detach().cpu())
                y_true.extend(lbl.numpy().tolist())

    if not all_logits:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    logits_np = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    y_np = np.asarray(y_true, dtype=np.int64)
    return logits_np, y_np


def find_best_threshold_binary(val_logits: np.ndarray, y_val: np.ndarray) -> tuple[float, dict]:
    if len(y_val) == 0:
        return 0.5, {"val_macro_f1": 0.0, "val_accuracy": 0.0}

    prob1 = np.exp(val_logits[:, 1] - np.max(val_logits, axis=1))
    prob0 = np.exp(val_logits[:, 0] - np.max(val_logits, axis=1))
    p = prob1 / (prob0 + prob1 + 1e-12)

    best_thr = 0.5
    best_f1 = -1.0
    best_acc = 0.0
    labels = [0, 1]
    for thr in np.linspace(0.05, 0.95, 91):
        pred = (p >= thr).astype(np.int64)
        f1 = f1_score(y_val, pred, labels=labels, average="macro", zero_division=0)
        acc = accuracy_score(y_val, pred)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_acc = float(acc)
            best_thr = float(thr)
    return best_thr, {"val_macro_f1": best_f1, "val_accuracy": best_acc}


def evaluate(model, loaders, device, class_names, out_dir, threshold: float | None = None):
    n_classes = len(class_names)
    labels = list(range(n_classes))
    logits_np, y_true = collect_logits_and_labels(model, loaders, device)
    if len(y_true) == 0:
        return {"accuracy_pct": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "n_test_samples": 0}

    logits = torch.from_numpy(logits_np)
    pred = logits_to_preds(logits, threshold=threshold).numpy().astype(np.int64)

    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=labels)
    report = classification_report(y_true, pred, labels=labels, target_names=class_names, digits=4, zero_division=0)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "classification_report_clean.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    np.savetxt(os.path.join(out_dir, "confusion_matrix_clean.csv"), cm, fmt="%d", delimiter=",")

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = "Confusion Matrix (clean)"
    if threshold is not None:
        title += f" | thr={threshold:.2f}"
    plt.title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_clean.png"), dpi=180)
    plt.close(fig)

    return {
        "accuracy_pct": float(acc * 100.0),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_test_samples": int(len(y_true)),
    }


def save_run_row(cfg: dict, metrics: dict, out_dir: str, threshold: float | None):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    row = {
        "experiment": str(cfg.get("project", {}).get("exp_id", cfg.get("project", {}).get("version", "unknown"))),
        "model": str(cfg.get("model", {}).get("name", "unknown")),
        "seed": int(train_cfg.get("seed", 42)),
        "window_seconds": float(data_cfg.get("window_seconds", 4.0)),
        "stride_seconds": float(data_cfg.get("stride_seconds", 1.0)),
        "accuracy_pct": metrics["accuracy_pct"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "n_test_samples": metrics["n_test_samples"],
        "decision_threshold": (float(threshold) if threshold is not None else "argmax"),
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


def _compute_label_config_nmi(split_summary: dict, cfg: dict) -> tuple[float, dict]:
    seg_by_test = {int(k): int(v) for k, v in split_summary.get("segments_by_test_id", {}).items()}
    total = int(sum(seg_by_test.values()))
    if total <= 0:
        return 0.0, {}

    label_map = {int(k): int(v) for k, v in cfg.get("data", {}).get("label_by_test_id", {}).items()}
    cfg_map = {int(k): str(v) for k, v in cfg.get("data", {}).get("test_config_by_id", {}).items()}

    joint = {}
    for tid, n in seg_by_test.items():
        y = int(label_map.get(tid, -1))
        c = str(cfg_map.get(tid, "NA"))
        joint[(c, y)] = joint.get((c, y), 0) + int(n)

    cfg_keys = sorted(set(c for c, _ in joint))
    label_keys = sorted(set(y for _, y in joint))
    pc = {c: sum(joint.get((c, y), 0) for y in label_keys) / total for c in cfg_keys}
    py = {y: sum(joint.get((c, y), 0) for c in cfg_keys) / total for y in label_keys}

    mi = 0.0
    for c in cfg_keys:
        for y in label_keys:
            pxy = joint.get((c, y), 0) / total
            if pxy > 0:
                mi += pxy * np.log2((pxy / (pc[c] * py[y] + 1e-12)) + 1e-12)
    hy = -sum(v * np.log2(v + 1e-12) for v in py.values())
    nmi = float(mi / (hy + 1e-12)) if hy > 0 else 0.0
    return nmi, joint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=".")
    parser.add_argument("--cfg_path", type=str, default="", help="Optional path to config json (overrides <exp_dir>/config.json).")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    cfg_path = os.path.abspath(args.cfg_path) if args.cfg_path else os.path.join(exp_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    if not os.path.isabs(cfg["data"]["data_dir"]):
        cfg["data"]["data_dir"] = os.path.abspath(os.path.join(exp_dir, cfg["data"]["data_dir"]))

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
    n_classes = int(cfg["data"]["num_classes"])

    log_dir = cfg["train"].get("log_dir", "./logs")
    model_dir = cfg["train"].get("model_dir", "./checkpoints")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(exp_dir, log_dir)
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(exp_dir, model_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_loaders, val_loaders, test_loaders, train_ds, val_ds, test_ds = create_dataloaders(cfg, return_datasets=True)
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Dataset split is empty. Please check data path, file naming, and split test IDs in config.json.")

    dataset_export_manifest(train_ds, os.path.join(log_dir, "manifest_train.csv"), split_name="train")
    dataset_export_manifest(val_ds, os.path.join(log_dir, "manifest_val.csv"), split_name="val")
    dataset_export_manifest(test_ds, os.path.join(log_dir, "manifest_test.csv"), split_name="test")
    train_summary = dataset_summary(train_ds, split_name="train")
    val_summary = dataset_summary(val_ds, split_name="val")
    test_summary = dataset_summary(test_ds, split_name="test")
    logging.info(f"Train summary: {train_summary}")
    logging.info(f"Val summary: {val_summary}")
    logging.info(f"Test summary: {test_summary}")

    train_cfg_nmi, train_joint = _compute_label_config_nmi(train_summary, cfg)
    logging.info(f"Train confound check: NMI(label,config)={train_cfg_nmi:.4f}, joint(config,label)={train_joint}")
    warn_thr = float(cfg["train"].get("warn_train_label_config_nmi", 0.85))
    fail_thr = float(cfg["train"].get("fail_train_label_config_nmi", 1.01))
    if train_cfg_nmi >= warn_thr:
        warnings.warn(
            f"High train confounding detected: NMI(label,config)={train_cfg_nmi:.4f}. "
            "Model may learn configuration shortcuts instead of damage-specific patterns.",
            RuntimeWarning,
        )
    if train_cfg_nmi >= fail_thr:
        raise RuntimeError(
            f"Train split confounding exceeds fail threshold: nmi={train_cfg_nmi:.4f} >= {fail_thr:.4f}. "
            "Please adjust split protocol."
        )

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg, device)

    opt_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"].get("learning_rate", 3e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    epochs = int(cfg["train"].get("epochs", 100))
    if bool(cfg["train"].get("use_cosine", True)):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    monitor_metric = str(cfg["train"].get("monitor_metric", "val_macro_f1")).lower()
    monitor_mode = "min" if monitor_metric == "val_loss" else "max"
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 0.0))

    scaler = GradScaler(enabled=amp_enabled)
    best_path = os.path.join(model_dir, cfg["train"].get("best_model_name", "best_model.pth"))
    stopper = EarlyStopping(
        patience=int(cfg["train"].get("early_stopping_patience", 20)),
        path=best_path,
        mode=monitor_mode,
        min_epochs=int(cfg["train"].get("early_stopping_min_epochs", 0)),
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_macro = run_epoch(
            model,
            train_loaders,
            criterion,
            device,
            n_classes=n_classes,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, val_acc, val_macro = run_epoch(
            model,
            val_loaders,
            criterion,
            device,
            n_classes=n_classes,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )
        if scheduler is not None:
            scheduler.step()

        calibrated_thr = None
        calibrated_val_acc = None
        if monitor_metric == "val_loss":
            monitor_value = val_loss
        elif monitor_metric == "val_acc":
            monitor_value = val_acc
        elif monitor_metric == "val_macro_f1_calibrated" and n_classes == 2:
            val_logits, y_val = collect_logits_and_labels(model, val_loaders, device)
            calibrated_thr, best_val = find_best_threshold_binary(val_logits, y_val)
            monitor_value = float(best_val["val_macro_f1"])
            calibrated_val_acc = float(best_val["val_accuracy"])
        else:
            monitor_value = val_macro

        stopper(monitor_value, model, epoch=epoch)

        if calibrated_thr is not None:
            logging.info(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_macro_f1={train_macro:.4f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro:.4f} "
                f"| calibrated_thr={calibrated_thr:.2f} calibrated_val_acc={calibrated_val_acc:.4f} "
                f"| monitor({monitor_metric})={monitor_value:.4f} | time={time.time()-t0:.1f}s"
            )
        else:
            logging.info(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_macro_f1={train_macro:.4f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro:.4f} "
                f"| monitor({monitor_metric})={monitor_value:.4f} | time={time.time()-t0:.1f}s"
            )

        if stopper.stop:
            logging.info("Early stopping triggered.")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    class_names = cfg["data"].get("class_names", ["Baseline", "Damaged"])
    threshold = None
    if bool(cfg["train"].get("calibrate_threshold", True)) and len(class_names) == 2:
        val_logits, y_val = collect_logits_and_labels(model, val_loaders, device)
        threshold, best_val = find_best_threshold_binary(val_logits, y_val)
        with open(os.path.join(log_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump({"threshold": threshold, **best_val}, f, ensure_ascii=False, indent=2)
        logging.info(f"Calibrated threshold on val: thr={threshold:.2f}, val_macro_f1={best_val['val_macro_f1']:.4f}, val_acc={best_val['val_accuracy']:.4f}")

    metrics = evaluate(model, test_loaders, device, class_names, log_dir, threshold=threshold)
    save_run_row(cfg, metrics, out_dir=log_dir, threshold=threshold)
    logging.info(f"Final clean metrics: {metrics}")


if __name__ == "__main__":
    main()
