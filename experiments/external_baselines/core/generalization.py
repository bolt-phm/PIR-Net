import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.dataset import create_dataloaders
from core.model import build_model


def evaluate_once(model, test_loaders, device, class_names):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for loader in test_loaders.values():
            for batch in loader:
                if batch is None:
                    continue
                img, sig, lbl = batch[:3]
                logits = model(img.to(device), sig.to(device))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(lbl.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, macro_f1, weighted_f1, rep, cm, len(y_true)


def save_csv_row(csv_path, row):
    Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=".")
    parser.add_argument("--snrs", nargs="+", default=["clean", "20", "10", "5", "0", "-5"])
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    with open(os.path.join(exp_dir, "config.json"), "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    device = torch.device(cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    log_dir = cfg["train"].get("log_dir", "./logs")
    model_dir = cfg["train"].get("model_dir", "./checkpoints")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(exp_dir, log_dir)
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(exp_dir, model_dir)

    model = build_model(cfg).to(device)
    best_path = os.path.join(model_dir, cfg["train"].get("best_model_name", "best_model.pth"))
    model.load_state_dict(torch.load(best_path, map_location=device))

    class_names = cfg["data"].get("case_ids", [str(i) for i in range(int(cfg["data"].get("num_classes", 6)))])

    csv_path = os.path.join(log_dir, "all_runs.csv")
    for snr in args.snrs:
        if str(snr).lower() == "clean":
            os.environ.pop("FORCE_SNR", None)
            snr_value = "clean"
            suffix = "clean"
        else:
            os.environ["FORCE_SNR"] = str(snr)
            snr_value = float(snr)
            suffix = f"snr{str(snr).replace('-', 'm').replace('.', 'p')}"

        _, _, test_loaders = create_dataloaders(cfg)
        acc, macro_f1, weighted_f1, report, cm, n_samples = evaluate_once(model, test_loaders, device, class_names)

        np.savetxt(os.path.join(log_dir, f"confusion_matrix_{suffix}.csv"), cm, fmt="%d", delimiter=",")
        with open(os.path.join(log_dir, f"classification_report_{suffix}.txt"), "w", encoding="utf-8-sig") as f:
            f.write(report)

        row = {
            "experiment": str(cfg.get("project", {}).get("exp_id", cfg.get("project", {}).get("version", "unknown"))),
            "model": str(cfg.get("model", {}).get("name", "unknown")),
            "split_mode": str(cfg["data"].get("split_mode", "temporal")),
            "seed": int(cfg["train"].get("seed", 42)),
            "signal_preprocess": str(cfg["data"].get("signal_preprocess", "linear")),
            "image_preprocess": str(cfg["data"].get("image_preprocess", "none")),
            "accuracy_pct": float(acc * 100.0),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "n_test_samples": int(n_samples),
            "snr_db": snr_value,
        }
        save_csv_row(csv_path, row)

    os.environ.pop("FORCE_SNR", None)


if __name__ == "__main__":
    main()
