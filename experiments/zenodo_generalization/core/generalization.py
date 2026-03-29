import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from core.dataset import create_test_loaders_only, make_eval_cfg
from core.model import build_model


def _set_eval_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_calibrated_threshold(log_dir: str) -> float | None:
    path = os.path.join(log_dir, "calibration.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        thr = data.get("threshold", None)
        if thr is None:
            return None
        thr_f = float(thr)
        if 0.0 < thr_f < 1.0:
            return thr_f
    except Exception:
        return None
    return None


def evaluate_once(model, test_loaders, device, class_names, threshold: float | None = None):
    y_true = []
    all_logits = []
    model.eval()
    with torch.inference_mode():
        for loader in test_loaders.values():
            for batch in loader:
                if batch is None:
                    continue
                img, sig, lbl = batch[:3]
                if img is not None:
                    img = img.to(device, non_blocking=True)
                sig = sig.to(device, non_blocking=True)
                logits = model(img, sig)
                all_logits.append(logits.detach().cpu())
                y_true.extend(lbl.numpy().tolist())

    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, "No samples.", np.zeros((len(class_names), len(class_names)), dtype=int), 0

    logits = torch.cat(all_logits, dim=0)
    if logits.size(1) == 2 and threshold is not None:
        prob_1 = torch.softmax(logits, dim=1)[:, 1]
        y_pred = (prob_1 >= float(threshold)).long().numpy().tolist()
    else:
        y_pred = torch.argmax(logits, dim=1).numpy().tolist()

    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, macro_f1, weighted_f1, rep, cm, len(y_true)


def save_csv_row(csv_path, row):
    Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _default_eval_groups():
    return {
        "iid_test_split": [12, 13, 14, 17],
        "axis_x_only": [1, 2, 3],
        "axis_y_only": [4, 5, 6, 7],
        "axis_xy_only": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "config_2_1_only": [11, 13, 16, 17],
        "config_2_2_only": [12, 14, 15],
    }


def _snr_to_suffix(snr) -> str:
    s = str(snr).lower()
    if s == "clean":
        return "clean"
    return f"snr{s.replace('-', 'm').replace('.', 'p')}"


def _stable_repeat_seed(base_seed: int, group: str, snr, rep_idx: int) -> int:
    tag = f"{group}|{snr}|{rep_idx}"
    h = int(hashlib.md5(tag.encode("utf-8")).hexdigest()[:8], 16)
    return int(base_seed + h % 1_000_003)


def _summarize_one(rows: list[dict]) -> dict:
    acc = np.asarray([float(r["accuracy_pct"]) for r in rows], dtype=np.float64)
    mf1 = np.asarray([float(r["macro_f1"]) for r in rows], dtype=np.float64)
    wf1 = np.asarray([float(r["weighted_f1"]) for r in rows], dtype=np.float64)
    ns = np.asarray([float(r["n_test_samples"]) for r in rows], dtype=np.float64)
    return {
        "accuracy_pct_mean": float(acc.mean()),
        "accuracy_pct_std": float(acc.std(ddof=0)),
        "macro_f1_mean": float(mf1.mean()),
        "macro_f1_std": float(mf1.std(ddof=0)),
        "weighted_f1_mean": float(wf1.mean()),
        "weighted_f1_std": float(wf1.std(ddof=0)),
        "n_test_samples_mean": float(ns.mean()),
        "n_repeats": int(len(rows)),
    }


def _check_monotonic_from_summary(summary_rows: list[dict]) -> list[dict]:
    out = []
    by_group: dict[str, dict[str, float]] = {}
    for r in summary_rows:
        g = str(r["group"])
        snr = str(r["snr_db"])
        by_group.setdefault(g, {})
        by_group[g][snr] = float(r["accuracy_pct_mean"])
    for g, d in by_group.items():
        if not {"clean", "10.0", "5.0", "0.0"}.issubset(set(d.keys())):
            continue
        clean = d["clean"]
        s10 = d["10.0"]
        s5 = d["5.0"]
        s0 = d["0.0"]
        non_monotonic = not (clean >= s10 >= s5 >= s0)
        out.append(
            {
                "group": g,
                "clean": clean,
                "snr10": s10,
                "snr5": s5,
                "snr0": s0,
                "is_monotonic_nonincreasing": (not non_monotonic),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=".")
    parser.add_argument("--cfg_path", type=str, default="", help="Optional path to config json (overrides <exp_dir>/config.json).")
    parser.add_argument("--group", type=str, default="all", help="Evaluate only one group key from data.eval_groups.")
    parser.add_argument("--snrs", nargs="+", default=["clean", "10", "5", "0"])
    parser.add_argument("--snr_repeats", type=int, default=1, help="Number of repeated evaluations for each noisy SNR.")
    parser.add_argument("--repeat_seed_base", type=int, default=-1, help="Base seed for repeated noisy evaluation. -1 uses train seed.")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    cfg_path = os.path.abspath(args.cfg_path) if args.cfg_path else os.path.join(exp_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    if not os.path.isabs(cfg["data"]["data_dir"]):
        cfg["data"]["data_dir"] = os.path.abspath(os.path.join(exp_dir, cfg["data"]["data_dir"]))

    device = torch.device(cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    log_dir = cfg["train"].get("log_dir", "./logs")
    model_dir = cfg["train"].get("model_dir", "./checkpoints")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(exp_dir, log_dir)
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(exp_dir, model_dir)
    os.makedirs(log_dir, exist_ok=True)

    raw_csv_path = os.path.join(log_dir, "generalization_runs.csv")
    summary_csv_path = os.path.join(log_dir, "generalization_summary.csv")
    if os.path.exists(raw_csv_path):
        os.remove(raw_csv_path)
    if os.path.exists(summary_csv_path):
        os.remove(summary_csv_path)

    threshold = None
    if bool(cfg["train"].get("use_calibrated_threshold_in_generalization", True)) and len(cfg["data"].get("class_names", ["Baseline", "Damaged"])) == 2:
        threshold = _load_calibrated_threshold(log_dir)

    model = build_model(cfg).to(device)
    best_path = os.path.join(model_dir, cfg["train"].get("best_model_name", "best_model.pth"))
    model.load_state_dict(torch.load(best_path, map_location=device))
    class_names = cfg["data"].get("class_names", ["Baseline", "Damaged"])

    eval_groups = cfg["data"].get("eval_groups", _default_eval_groups())
    if args.group != "all":
        if args.group not in eval_groups:
            raise KeyError(f"Unknown group '{args.group}'. Available: {sorted(eval_groups.keys())}")
        eval_groups = {args.group: eval_groups[args.group]}
    if bool(cfg["data"].get("skip_single_class_groups_in_generalization", True)):
        label_map = {int(k): int(v) for k, v in cfg["data"].get("label_by_test_id", {}).items()}
        kept = {}
        for g_name, g_ids in eval_groups.items():
            labels = {label_map.get(int(t), -1) for t in g_ids}
            if len([x for x in labels if x >= 0]) < 2:
                continue
            kept[g_name] = g_ids
        eval_groups = kept
    if not eval_groups:
        raise RuntimeError("No valid eval groups remain after filtering. Please define at least one group containing both classes.")

    summary_rows = []
    base_seed = int(cfg["train"].get("seed", 42)) if int(args.repeat_seed_base) < 0 else int(args.repeat_seed_base)
    prev_cache_silent = os.environ.get("RAM_CACHE_SILENT")
    os.environ["RAM_CACHE_SILENT"] = "1"
    for g_name, g_ids in eval_groups.items():
        g_cfg = make_eval_cfg(cfg, [int(x) for x in g_ids])
        for snr in args.snrs:
            snr_text = str(snr).lower()
            is_clean = snr_text == "clean"
            snr_value = "clean" if is_clean else float(snr)
            repeats = 1 if is_clean else max(1, int(args.snr_repeats))
            per_setting_rows: list[dict] = []

            if is_clean:
                os.environ.pop("FORCE_SNR", None)
            else:
                os.environ["FORCE_SNR"] = str(snr)
            test_loaders = create_test_loaders_only(g_cfg)

            for rep_idx in range(repeats):
                if is_clean:
                    os.environ.pop("FORCE_SNR", None)
                else:
                    os.environ["FORCE_SNR"] = str(snr)
                eval_seed = _stable_repeat_seed(base_seed, g_name, snr_value, rep_idx)
                _set_eval_seed(eval_seed)
                acc, macro_f1, weighted_f1, report, cm, n_samples = evaluate_once(
                    model,
                    test_loaders,
                    device,
                    class_names,
                    threshold=threshold,
                )

                suffix = _snr_to_suffix(snr_value)
                if repeats > 1:
                    suffix = f"{suffix}_r{rep_idx+1}"
                np.savetxt(os.path.join(log_dir, f"cm_{g_name}_{suffix}.csv"), cm, fmt="%d", delimiter=",")
                with open(os.path.join(log_dir, f"report_{g_name}_{suffix}.txt"), "w", encoding="utf-8") as f:
                    f.write(report)

                row = {
                    "experiment": str(cfg.get("project", {}).get("exp_id", cfg.get("project", {}).get("version", "unknown"))),
                    "group": g_name,
                    "test_ids": ",".join(str(int(x)) for x in g_ids),
                    "model": str(cfg.get("model", {}).get("name", "unknown")),
                    "seed": int(cfg["train"].get("seed", 42)),
                    "repeat_idx": int(rep_idx + 1),
                    "repeat_seed": int(eval_seed),
                    "accuracy_pct": float(acc * 100.0),
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(weighted_f1),
                    "n_test_samples": int(n_samples),
                    "snr_db": snr_value,
                    "decision_threshold": (float(threshold) if threshold is not None else "argmax"),
                }
                save_csv_row(raw_csv_path, row)
                per_setting_rows.append(row)

            sm = _summarize_one(per_setting_rows)
            summary_row = {
                "experiment": str(cfg.get("project", {}).get("exp_id", cfg.get("project", {}).get("version", "unknown"))),
                "group": g_name,
                "test_ids": ",".join(str(int(x)) for x in g_ids),
                "model": str(cfg.get("model", {}).get("name", "unknown")),
                "seed": int(cfg["train"].get("seed", 42)),
                "snr_db": snr_value,
                "decision_threshold": (float(threshold) if threshold is not None else "argmax"),
                **sm,
            }
            save_csv_row(summary_csv_path, summary_row)
            summary_rows.append(summary_row)
            del test_loaders

    quality = {
        "snr_repeats": int(args.snr_repeats),
        "repeat_seed_base": int(base_seed),
        "monotonic_checks": _check_monotonic_from_summary(summary_rows),
    }
    with open(os.path.join(log_dir, "generalization_quality_checks.json"), "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)

    os.environ.pop("FORCE_SNR", None)
    if prev_cache_silent is None:
        os.environ.pop("RAM_CACHE_SILENT", None)
    else:
        os.environ["RAM_CACHE_SILENT"] = prev_cache_silent


if __name__ == "__main__":
    main()
