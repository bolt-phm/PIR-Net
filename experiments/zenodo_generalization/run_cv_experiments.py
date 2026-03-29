import argparse
import csv
import json
import os
import shutil
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


def _model_needs_image(cfg: dict) -> bool:
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "")).lower()
    if model_name in {"pirnet_dual_branch", "pirnet", "pirnet_lite", "pgrf_dual_branch", "pgrfnet", "pgrf-net"}:
        branch_mode = str(model_cfg.get("branch_mode", "dual")).lower()
        return branch_mode in {"dual", "image_only"}
    return False


def _load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_row(csv_path: Path, row: dict):
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

    if rows:
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

    # Aggregate noisy generalization summary over folds x seeds.
    g_rows = []
    for p in sorted(exp_dir.glob("logs/cv_*/*/generalization_summary.csv")):
        fold = p.parent.parent.name.replace("cv_", "")
        seed = p.parent.name.replace("seed_", "")
        try:
            with p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    g_rows.append(
                        {
                            "fold": fold,
                            "seed": int(seed),
                            "group": row.get("group", ""),
                            "snr_db": str(row.get("snr_db", "")),
                            "accuracy_pct_mean": float(row.get("accuracy_pct_mean", 0.0)),
                            "accuracy_pct_std": float(row.get("accuracy_pct_std", 0.0)),
                            "macro_f1_mean": float(row.get("macro_f1_mean", 0.0)),
                            "macro_f1_std": float(row.get("macro_f1_std", 0.0)),
                            "weighted_f1_mean": float(row.get("weighted_f1_mean", 0.0)),
                            "weighted_f1_std": float(row.get("weighted_f1_std", 0.0)),
                            "n_repeats": int(float(row.get("n_repeats", 1))),
                            "source_csv": str(p),
                        }
                    )
        except Exception:
            continue

    if g_rows:
        g_all_path = exp_dir / "logs" / "cv_generalization_all_runs.csv"
        with g_all_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(g_rows[0].keys()))
            w.writeheader()
            w.writerows(g_rows)

        grouped = {}
        for r in g_rows:
            key = (r["group"], r["snr_db"])
            grouped.setdefault(key, [])
            grouped[key].append(r)

        g_summary_rows = []
        for (group, snr), arr in sorted(grouped.items(), key=lambda x: (x[0][0], str(x[0][1]))):
            acc = np.asarray([x["accuracy_pct_mean"] for x in arr], dtype=np.float64)
            mf1 = np.asarray([x["macro_f1_mean"] for x in arr], dtype=np.float64)
            wf1 = np.asarray([x["weighted_f1_mean"] for x in arr], dtype=np.float64)
            g_summary_rows.append(
                {
                    "group": group,
                    "snr_db": snr,
                    "n_fold_seed_runs": int(len(arr)),
                    "accuracy_pct_mean": float(acc.mean()),
                    "accuracy_pct_std": float(acc.std(ddof=0)),
                    "macro_f1_mean": float(mf1.mean()),
                    "macro_f1_std": float(mf1.std(ddof=0)),
                    "weighted_f1_mean": float(wf1.mean()),
                    "weighted_f1_std": float(wf1.std(ddof=0)),
                }
            )

        g_summary_path = exp_dir / "logs" / "cv_generalization_summary.csv"
        with g_summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(g_summary_rows[0].keys()))
            w.writeheader()
            w.writerows(g_summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["401", "402", "403", "404", "501", "502", "503"])
    parser.add_argument("--folds", nargs="*", default=[], help="Optional subset of folds, e.g. fold_d15_cfg22 fold_d16_cfg21.")
    parser.add_argument("--with_generalization", action="store_true")
    parser.add_argument(
        "--generalization_experiments",
        nargs="*",
        default=[],
        help="Optional subset for noisy generalization. If empty, all experiments listed in --experiments will run generalization.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[], help="Optional list of seeds. Defaults to config seed.")
    parser.add_argument("--snrs", nargs="+", default=["clean", "10", "5", "0"])
    parser.add_argument("--snr_repeats", type=int, default=1)
    parser.add_argument("--eval_workers", type=int, default=0, help="num_workers for val/test dataloaders during C-protocol runs.")
    parser.add_argument("--num_workers_train_override", type=int, default=-1, help="Override data.num_workers_train for all runs if >= 0.")
    parser.add_argument("--batch_size_train_override", type=int, default=0, help="Override data.batch_size_train for all runs if > 0.")
    parser.add_argument("--batch_size_eval_override", type=int, default=0, help="Override data.batch_size_eval for all runs if > 0.")
    parser.add_argument("--prefetch_factor_override", type=int, default=0, help="Override data.prefetch_factor for all runs if > 0.")
    parser.add_argument("--enable_ram_cache", action="store_true", help="Enable RAM cache in generated fold configs.")
    parser.add_argument("--ram_cache_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--ram_cache_train_image", action="store_true", help="Cache train pseudo-images (speed-first).")
    parser.add_argument(
        "--ram_cache_allow_aug_mismatch",
        action="store_true",
        help="Allow train image cache even when signal augmentation is enabled.",
    )
    parser.add_argument("--main_experiment", type=str, default="501", help="Experiment ID treated as the main model for patience override.")
    parser.add_argument("--patience_main", type=int, default=25, help="Early-stopping patience for main model.")
    parser.add_argument("--patience_other", type=int, default=10, help="Early-stopping patience for all non-main models.")
    parser.add_argument("--min_epochs_main", type=int, default=15, help="Minimum epochs before early-stop for main model.")
    parser.add_argument("--min_epochs_other", type=int, default=8, help="Minimum epochs before early-stop for non-main models.")
    parser.add_argument("--clean_existing", action="store_true", help="Remove existing cv logs/checkpoints before running.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip completed fold/seed runs. If metrics exist but generalization summary is missing, runs generalization only.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    folds = _default_folds()
    if args.folds:
        wanted = set(str(x) for x in args.folds)
        folds = {k: v for k, v in folds.items() if k in wanted}
        if not folds:
            raise ValueError(f"No valid folds selected from {args.folds}. Available: {list(_default_folds().keys())}")
    gen_subset = {str(x) for x in (args.generalization_experiments or [])}

    for exp in args.experiments:
        exp_dir = root / exp
        cfg_path = exp_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config: {cfg_path}")
        base_cfg = _load_json(cfg_path)
        all_ids = sorted(int(x) for x in base_cfg["data"]["label_by_test_id"].keys())
        seeds = args.seeds[:] if args.seeds else [int(base_cfg["train"].get("seed", 42))]

        if args.clean_existing:
            for p in [exp_dir / "logs", exp_dir / "checkpoints", exp_dir / ".cv_configs"]:
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)

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
                cfg["data"]["num_workers_eval"] = int(args.eval_workers)
                if int(args.num_workers_train_override) >= 0:
                    cfg["data"]["num_workers_train"] = int(args.num_workers_train_override)
                if int(args.batch_size_train_override) > 0:
                    cfg["data"]["batch_size_train"] = int(args.batch_size_train_override)
                if int(args.batch_size_eval_override) > 0:
                    cfg["data"]["batch_size_eval"] = int(args.batch_size_eval_override)
                if int(args.prefetch_factor_override) > 0:
                    cfg["data"]["prefetch_factor"] = int(args.prefetch_factor_override)
                needs_image = _model_needs_image(cfg)
                if not needs_image:
                    cfg["data"]["use_pseudo_image"] = False
                    cfg["data"]["omit_image_tensor_when_disabled"] = True
                else:
                    cfg["data"]["use_pseudo_image"] = bool(cfg["data"].get("use_pseudo_image", True))
                    cfg["data"]["omit_image_tensor_when_disabled"] = False
                if bool(args.enable_ram_cache):
                    cfg["data"]["ram_cache"] = {
                        "enabled": True,
                        "dtype": str(args.ram_cache_dtype),
                        "cache_train_signal": True,
                        "cache_train_image": bool(args.ram_cache_train_image and needs_image),
                        "cache_eval_signal": True,
                        "cache_eval_image": bool(needs_image),
                        "allow_train_image_cache_with_augmentation": bool(args.ram_cache_allow_aug_mismatch),
                        "log_interval": 0,
                    }
                cfg["data"]["eval_groups"] = {
                    "fold_test": test_ids,
                    "config_2_1_mix": _safe_group([11, 13, 16, 17], set(all_ids)),
                    "config_2_2_mix": _safe_group([12, 14, 15], set(all_ids)),
                }
                cfg["train"]["seed"] = int(seed)
                is_main = str(exp) == str(args.main_experiment)
                cfg["train"]["early_stopping_patience"] = int(args.patience_main if is_main else args.patience_other)
                cfg["train"]["early_stopping_min_epochs"] = int(args.min_epochs_main if is_main else args.min_epochs_other)
                cfg["train"]["log_dir"] = f"./logs/cv_{fold_name}/seed_{seed}"
                cfg["train"]["model_dir"] = f"./checkpoints/cv_{fold_name}/seed_{seed}"
                cfg["train"]["warn_train_label_config_nmi"] = 0.8
                cfg["train"]["fail_train_label_config_nmi"] = 1.01

                fold_cfg_path = exp_dir / ".cv_configs" / f"{fold_name}_seed{seed}.json"
                _save_json(fold_cfg_path, cfg)

                do_generalization = bool(args.with_generalization) and (len(gen_subset) == 0 or str(exp) in gen_subset)
                metrics_path = exp_dir / cfg["train"]["log_dir"].replace("./", "") / "metrics_clean.json"
                gen_summary_path = exp_dir / cfg["train"]["log_dir"].replace("./", "") / "generalization_summary.csv"

                run_train = True
                run_generalization = do_generalization
                if args.skip_existing and metrics_path.exists():
                    if do_generalization and gen_summary_path.exists():
                        print(f"[SKIP] completed fold={fold_name} seed={seed} (metrics + generalization exist).")
                        continue
                    run_train = False
                    if do_generalization and not gen_summary_path.exists():
                        print(f"[SKIP] train exists; run generalization only for fold={fold_name} seed={seed}.")
                        run_generalization = True
                    elif not do_generalization:
                        print(f"[SKIP] train exists; generalization disabled for exp={exp}.")
                        continue

                if run_train:
                    run_cmd([py, "train.py", "--exp_dir", str(exp_dir), "--cfg_path", str(fold_cfg_path)], cwd=str(exp_dir))
                if run_generalization:
                    run_cmd(
                        [
                            py,
                            "generalization.py",
                            "--exp_dir",
                            str(exp_dir),
                            "--cfg_path",
                            str(fold_cfg_path),
                            "--snr_repeats",
                            str(int(args.snr_repeats)),
                            "--snrs",
                            *[str(x) for x in args.snrs],
                        ],
                        cwd=str(exp_dir),
                    )
                elif args.with_generalization and len(gen_subset) > 0 and str(exp) not in gen_subset:
                    print(f"[SKIP] generalization disabled for exp={exp} (not in --generalization_experiments).")

                if metrics_path.exists():
                    m = _load_json(metrics_path)
                    _append_row(
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
