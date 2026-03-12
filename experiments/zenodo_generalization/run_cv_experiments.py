import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_cmd(cmd, cwd):
    print(f"[RUN] {' '.join(cmd)} (cwd={cwd})")
    subprocess.check_call(cmd, cwd=cwd)


def _default_folds():
    return {
        "fold_d15_cfg22": [12, 14, 15],
        "fold_d16_cfg21": [11, 13, 16],
        "fold_d17_cfg21": [11, 13, 17],
    }


def _safe_group(ids, allowed):
    return [int(x) for x in ids if int(x) in allowed]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_cv_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _summarize_cv(exp_dir: Path):
    rows = []
    for p in sorted(exp_dir.glob("logs/cv_*/*/metrics_clean.json")):
        try:
            d = _load_json(p)
        except Exception:
            continue
        rows.append(
            {
                "path": str(p),
                "experiment": d.get("experiment"),
                "model": d.get("model"),
                "seed": d.get("seed"),
                "accuracy_pct": float(d.get("accuracy_pct", 0.0)),
                "macro_f1": float(d.get("macro_f1", 0.0)),
                "weighted_f1": float(d.get("weighted_f1", 0.0)),
                "decision_threshold": d.get("decision_threshold", "argmax"),
            }
        )

    if not rows:
        return

    out_csv = exp_dir / "logs" / "cv_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    accs = np.asarray([r["accuracy_pct"] for r in rows], dtype=np.float64)
    f1s = np.asarray([r["macro_f1"] for r in rows], dtype=np.float64)
    summary = {
        "n_runs": int(len(rows)),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std(ddof=0)),
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std(ddof=0)),
    }
    (exp_dir / "logs" / "cv_summary_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["401", "402", "403", "404"])
    parser.add_argument("--with_generalization", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[], help="Optional list of seeds. Defaults to config seed.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    folds = _default_folds()

    for exp in args.experiments:
        exp_dir = root / exp
        cfg_path = exp_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config: {cfg_path}")
        base_cfg = _load_json(cfg_path)
        all_ids = sorted(int(x) for x in base_cfg["data"]["label_by_test_id"].keys())
        seeds = args.seeds[:] if args.seeds else [int(base_cfg["train"].get("seed", 42))]

        for fold_name, test_ids_raw in folds.items():
            test_ids = _safe_group(test_ids_raw, set(all_ids))
            if len(test_ids) == 0:
                continue
            train_ids = [x for x in all_ids if x not in set(test_ids)]
            for seed in seeds:
                cfg = json.loads(json.dumps(base_cfg))
                cfg["project"]["version"] = f"{base_cfg['project'].get('version', 'exp')}-{fold_name}-s{seed}"
                cfg["project"]["exp_id"] = f"{exp}_{fold_name}_s{seed}"
                cfg["data"]["train_test_ids"] = train_ids
                cfg["data"]["val_test_ids"] = []
                cfg["data"]["test_test_ids"] = test_ids
                cfg["data"]["use_val_from_train"] = True
                cfg["data"]["val_from_train_ratio"] = float(args.val_ratio)
                cfg["data"]["val_from_train_seed"] = int(seed)
                cfg["data"]["skip_single_class_groups_in_generalization"] = True
                cfg["data"]["eval_groups"] = {
                    "fold_test": test_ids,
                    "config_2_1_mix": _safe_group([11, 13, 16, 17], set(all_ids)),
                    "config_2_2_mix": _safe_group([12, 14, 15], set(all_ids)),
                }
                cfg["train"]["seed"] = int(seed)
                cfg["train"]["log_dir"] = f"./logs/cv_{fold_name}/seed_{seed}"
                cfg["train"]["model_dir"] = f"./checkpoints/cv_{fold_name}/seed_{seed}"
                cfg["train"]["warn_train_label_config_nmi"] = 0.8
                cfg["train"]["fail_train_label_config_nmi"] = 1.01

                fold_cfg_path = exp_dir / ".cv_configs" / f"{fold_name}_seed{seed}.json"
                _save_json(fold_cfg_path, cfg)

                run_cmd([py, "train.py", "--exp_dir", str(exp_dir), "--cfg_path", str(fold_cfg_path)], cwd=str(exp_dir))
                if args.with_generalization:
                    run_cmd([py, "generalization.py", "--exp_dir", str(exp_dir), "--cfg_path", str(fold_cfg_path)], cwd=str(exp_dir))

                metrics_path = exp_dir / cfg["train"]["log_dir"].replace("./", "") / "metrics_clean.json"
                if metrics_path.exists():
                    m = _load_json(metrics_path)
                    _append_cv_row(
                        exp_dir / "logs" / "cv_all_runs.csv",
                        {
                            "experiment": exp,
                            "fold": fold_name,
                            "seed": seed,
                            "accuracy_pct": m.get("accuracy_pct", 0.0),
                            "macro_f1": m.get("macro_f1", 0.0),
                            "weighted_f1": m.get("weighted_f1", 0.0),
                            "decision_threshold": m.get("decision_threshold", "argmax"),
                            "metrics_path": str(metrics_path),
                        },
                    )

        _summarize_cv(exp_dir)


if __name__ == "__main__":
    main()
