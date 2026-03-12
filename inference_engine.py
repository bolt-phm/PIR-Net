import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def log(msg: str) -> None:
    print(f"LOG:{msg}", flush=True)


def _load_runtime_modules():
    from dataset import generate_pseudo_image, smart_resample
    from generalization import EnsembleModel, evaluate, load_config, resolve_model_paths
    return generate_pseudo_image, smart_resample, EnsembleModel, evaluate, load_config, resolve_model_paths


def load_best_model(cfg: dict, exp_dir: str, use_ensemble: bool, device: torch.device, EnsembleModel, resolve_model_paths):
    cfg_local = copy.deepcopy(cfg)
    model_paths = resolve_model_paths(cfg_local, exp_dir)
    if not use_ensemble and len(model_paths) > 1:
        model_paths = model_paths[:1]

    if not use_ensemble:
        cfg_local.setdefault("inference", {})
        cfg_local["inference"]["ensemble_models"] = model_paths

    model = EnsembleModel(cfg_local, model_paths, device)
    model.eval()
    return model


def find_sample_npy(cfg: dict, exp_dir: str, user_path: str | None) -> str:
    if user_path:
        p = user_path if os.path.isabs(user_path) else os.path.join(exp_dir, user_path)
        if os.path.exists(p):
            return p

    data_dir = cfg["data"].get("data_dir", "")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(exp_dir, data_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    for fp in Path(data_dir).rglob("*.npy"):
        return str(fp)

    raise FileNotFoundError(f"No .npy files found under: {data_dir}")


def build_single_input(raw_signal: np.ndarray, cfg: dict, smart_resample, generate_pseudo_image):
    target_len = int(cfg["data"].get("target_signal_len", 8192))
    factor = int(cfg["data"].get("base_resample_factor", 150))
    alpha = float(cfg["data"].get("resample_alpha", 0.7))
    global_scale = float(cfg["data"].get("global_scale", 1.0))

    needed = target_len * factor
    if len(raw_signal) < needed:
        chunk = raw_signal
    else:
        start = max(0, (len(raw_signal) - needed) // 2)
        chunk = raw_signal[start : start + needed]

    sig = smart_resample(chunk, target_len=target_len, alpha=alpha)
    sig = sig - float(np.mean(sig))
    sig = sig / (global_scale + 1e-6)

    if bool(cfg.get("data", {}).get("disable_pseudo_image", False)):
        h, w = tuple(cfg["data"].get("image_size", [224, 224]))
        img = np.zeros((h, w, 5), dtype=np.float32)
    else:
        img = generate_pseudo_image(sig, cfg)

    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    sig_t = torch.from_numpy(sig).float().unsqueeze(0).unsqueeze(0)
    return img_t, sig_t


def run_usb_mode(args):
    generate_pseudo_image, smart_resample, EnsembleModel, _, load_config, resolve_model_paths = _load_runtime_modules()

    exp_dir = os.path.abspath(args.exp_dir)
    cfg = load_config(exp_dir)
    device = torch.device(cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    model = load_best_model(cfg, exp_dir, use_ensemble=args.ensemble, device=device, EnsembleModel=EnsembleModel, resolve_model_paths=resolve_model_paths)

    sample_path = find_sample_npy(cfg, exp_dir, args.sample_npy)
    log(f"Using sample file: {sample_path}")

    raw = np.load(sample_path, mmap_mode="r")
    raw = np.asarray(raw, dtype=np.float32)

    img_t, sig_t = build_single_input(raw, cfg, smart_resample=smart_resample, generate_pseudo_image=generate_pseudo_image)
    img_t = img_t.to(device)
    sig_t = sig_t.to(device)

    with torch.no_grad():
        logits = model(img_t, sig_t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    pred_idx = int(pred.item())
    confidence = float(conf.item())
    class_names = cfg["data"].get("case_ids", [str(i) for i in range(int(cfg["data"].get("num_classes", 6)))])
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    payload = {
        "status": "success",
        "prediction": pred_name,
        "confidence": confidence,
        "index": pred_idx,
        "sample_file": os.path.basename(sample_path),
        "mode": "usb-sim",
    }
    print("JSON_RESULT:" + json.dumps(payload, ensure_ascii=False), flush=True)


def run_generalization_mode(args):
    _, _, _, evaluate, _, _ = _load_runtime_modules()

    exp_dir = os.path.abspath(args.exp_dir)
    log("Running generalization evaluation...")
    res = evaluate(exp_dir=exp_dir, force_snr=args.snr_db, save_artifacts=True)

    cm_src = res.get("confusion_matrix_path", "")
    if cm_src and os.path.exists(cm_src):
        log_dir = os.path.dirname(cm_src)
        dst = os.path.join(log_dir, "final_generalization_matrix.png")
        shutil.copy2(cm_src, dst)
        log(f"Saved GUI matrix image: {dst}")

    log(f"Accuracy={res.get('accuracy', 0.0):.4f}, Samples={res.get('n_test_samples', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Bridge script for BoltDetectionGUI")
    parser.add_argument("--mode", choices=["usb", "generalization"], required=True)
    parser.add_argument("--exp_dir", type=str, default=".")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--sample_npy", type=str, default="")
    parser.add_argument("--snr_db", type=float, default=None)
    args = parser.parse_args()

    try:
        if args.mode == "usb":
            run_usb_mode(args)
        else:
            run_generalization_mode(args)
    except Exception as e:
        payload = {
            "status": "error",
            "message": str(e),
            "mode": args.mode,
        }
        print("JSON_RESULT:" + json.dumps(payload, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    main()
