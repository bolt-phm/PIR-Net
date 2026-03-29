#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


TRAIN_EXPERIMENTS = ["401", "402", "403", "404", "501", "502", "503"]
GENERALIZATION_EXPERIMENTS = ["404", "501"]
FOLDS = ["d15_cfg22", "d16_cfg21", "d17_cfg21"]
SEEDS = ["2026", "3407", "4096"]

TRAIN_RE = re.compile(
    r"zenodo_experiment[\\/](\d+)[\\/]logs[\\/]cv_fold_(d\d+_cfg\d+)[\\/]seed_(\d+)[\\/]metrics_clean\.json$"
)
GEN_RE = re.compile(
    r"zenodo_experiment[\\/](\d+)[\\/]logs[\\/]cv_fold_(d\d+_cfg\d+)[\\/]seed_(\d+)[\\/]generalization_summary\.csv$"
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def expected_keys(experiments: list[str]) -> set[tuple[str, str, str]]:
    return {(e, fold, seed) for e in experiments for fold in FOLDS for seed in SEEDS}


def pick_preferred(existing: dict | None, candidate: dict) -> dict:
    if existing is None:
        return candidate
    old_mtime = existing["path"].stat().st_mtime
    new_mtime = candidate["path"].stat().st_mtime
    return candidate if new_mtime > old_mtime else existing


def scan_sources(source_root: Path, servers: list[str]) -> tuple[dict, dict]:
    train_map: dict[tuple[str, str, str], dict] = {}
    gen_map: dict[tuple[str, str, str], dict] = {}
    for server in servers:
        zroot = source_root / server / "zenodo_experiment"
        if not zroot.exists():
            continue

        for p in zroot.rglob("metrics_clean.json"):
            m = TRAIN_RE.search(str(p))
            if not m:
                continue
            exp, fold, seed = m.group(1), m.group(2), m.group(3)
            if exp not in TRAIN_EXPERIMENTS or fold not in FOLDS or seed not in SEEDS:
                continue
            key = (exp, fold, seed)
            candidate = {"server": server, "path": p}
            train_map[key] = pick_preferred(train_map.get(key), candidate)

        for p in zroot.rglob("generalization_summary.csv"):
            m = GEN_RE.search(str(p))
            if not m:
                continue
            exp, fold, seed = m.group(1), m.group(2), m.group(3)
            if exp not in GENERALIZATION_EXPERIMENTS or fold not in FOLDS or seed not in SEEDS:
                continue
            key = (exp, fold, seed)
            candidate = {"server": server, "path": p}
            gen_map[key] = pick_preferred(gen_map.get(key), candidate)
    return train_map, gen_map


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge A-F round-2 Zenodo results into tracked repository artifacts.")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source_root", type=Path, default=Path(r"E:\Desktop\GPT_Codex\2modify\huichuan"))
    parser.add_argument("--servers", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    parser.add_argument("--output_rel", default="experiment_results/zenodo_round2_cv_63x18")
    args = parser.parse_args()

    repo_root: Path = args.repo_root.resolve()
    source_root: Path = args.source_root.resolve()
    out_root: Path = (repo_root / args.output_rel).resolve()

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_map, gen_map = scan_sources(source_root, args.servers)
    expected_train = expected_keys(TRAIN_EXPERIMENTS)
    expected_gen = expected_keys(GENERALIZATION_EXPERIMENTS)

    missing_train = sorted(expected_train - set(train_map))
    missing_gen = sorted(expected_gen - set(gen_map))

    train_rows: list[dict] = []
    train_grouped: dict[str, list[dict]] = defaultdict(list)

    for exp, fold, seed in sorted(train_map):
        src = train_map[(exp, fold, seed)]["path"]
        server = train_map[(exp, fold, seed)]["server"]
        dst = out_root / "raw" / exp / f"cv_fold_{fold}" / f"seed_{seed}" / "metrics_clean.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        d = load_json(dst)
        row = {
            "experiment": exp,
            "fold": fold,
            "seed": seed,
            "accuracy_pct": safe_float(d.get("accuracy_pct")),
            "macro_f1": safe_float(d.get("macro_f1")),
            "weighted_f1": safe_float(d.get("weighted_f1")),
            "n_test_samples": int(safe_float(d.get("n_test_samples", 0))),
            "decision_threshold": d.get("decision_threshold", ""),
            "source_server": server,
            "source_path": str(src).replace("\\", "/"),
            "copied_path": str(dst.relative_to(out_root)).replace("\\", "/"),
            "sha256": sha256_of(dst),
        }
        train_rows.append(row)
        train_grouped[exp].append(row)

    gen_file_rows: list[dict] = []
    gen_long_rows: list[dict] = []

    for exp, fold, seed in sorted(gen_map):
        src = gen_map[(exp, fold, seed)]["path"]
        server = gen_map[(exp, fold, seed)]["server"]
        dst = out_root / "raw" / exp / f"cv_fold_{fold}" / f"seed_{seed}" / "generalization_summary.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        gen_file_rows.append(
            {
                "experiment": exp,
                "fold": fold,
                "seed": seed,
                "source_server": server,
                "source_path": str(src).replace("\\", "/"),
                "copied_path": str(dst.relative_to(out_root)).replace("\\", "/"),
                "sha256": sha256_of(dst),
            }
        )
        with dst.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gen_long_rows.append(
                    {
                        "experiment": exp,
                        "fold": fold,
                        "seed": seed,
                        "group": r.get("group", ""),
                        "snr_db": str(r.get("snr_db", "")),
                        "accuracy_pct_mean": safe_float(r.get("accuracy_pct_mean", 0.0)),
                        "accuracy_pct_std": safe_float(r.get("accuracy_pct_std", 0.0)),
                        "macro_f1_mean": safe_float(r.get("macro_f1_mean", 0.0)),
                        "macro_f1_std": safe_float(r.get("macro_f1_std", 0.0)),
                        "weighted_f1_mean": safe_float(r.get("weighted_f1_mean", 0.0)),
                        "weighted_f1_std": safe_float(r.get("weighted_f1_std", 0.0)),
                        "n_repeats": int(safe_float(r.get("n_repeats", 0))),
                    }
                )

    train_summary_rows: list[dict] = []
    for exp in sorted(train_grouped):
        rows = train_grouped[exp]
        acc = [r["accuracy_pct"] for r in rows]
        mf1 = [r["macro_f1"] for r in rows]
        wf1 = [r["weighted_f1"] for r in rows]
        train_summary_rows.append(
            {
                "experiment": exp,
                "n_runs": len(rows),
                "accuracy_pct_mean": mean(acc),
                "accuracy_pct_std": pstdev(acc) if len(acc) > 1 else 0.0,
                "macro_f1_mean": mean(mf1),
                "macro_f1_std": pstdev(mf1) if len(mf1) > 1 else 0.0,
                "weighted_f1_mean": mean(wf1),
                "weighted_f1_std": pstdev(wf1) if len(wf1) > 1 else 0.0,
                "best_accuracy_pct": max(acc),
                "worst_accuracy_pct": min(acc),
            }
        )

    grouped_gen: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in gen_long_rows:
        grouped_gen[(r["experiment"], r["group"], r["snr_db"])].append(r)

    gen_summary_rows: list[dict] = []
    for (exp, group, snr), rows in sorted(grouped_gen.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        acc = [x["accuracy_pct_mean"] for x in rows]
        mf1 = [x["macro_f1_mean"] for x in rows]
        wf1 = [x["weighted_f1_mean"] for x in rows]
        gen_summary_rows.append(
            {
                "experiment": exp,
                "group": group,
                "snr_db": snr,
                "n_fold_seed_runs": len(rows),
                "accuracy_pct_mean": mean(acc),
                "accuracy_pct_std": pstdev(acc) if len(acc) > 1 else 0.0,
                "macro_f1_mean": mean(mf1),
                "macro_f1_std": pstdev(mf1) if len(mf1) > 1 else 0.0,
                "weighted_f1_mean": mean(wf1),
                "weighted_f1_std": pstdev(wf1) if len(wf1) > 1 else 0.0,
            }
        )

    write_csv(out_root / "train_manifest.csv", train_rows)
    write_csv(out_root / "train_summary_by_experiment.csv", train_summary_rows)
    write_csv(out_root / "generalization_file_manifest.csv", gen_file_rows)
    write_csv(out_root / "generalization_long.csv", gen_long_rows)
    write_csv(out_root / "generalization_summary_by_group_snr.csv", gen_summary_rows)

    coverage = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root).replace("\\", "/"),
        "servers": args.servers,
        "expected_train_runs": len(expected_train),
        "found_train_runs": len(train_map),
        "missing_train_runs": [list(x) for x in missing_train],
        "expected_generalization_runs": len(expected_gen),
        "found_generalization_runs": len(gen_map),
        "missing_generalization_runs": [list(x) for x in missing_gen],
        "is_train_complete": len(missing_train) == 0,
        "is_generalization_complete": len(missing_gen) == 0,
    }
    write_json(out_root / "coverage_summary.json", coverage)

    readme = f"""# Zenodo Round-2 CV Result Package (63+18)

This folder contains the merged round-2 result package for Zenodo cross-condition experiments, aggregated from servers `{", ".join(args.servers)}`.

## Scope

- Clean CV train/evaluation runs: 7 experiments x 3 folds x 3 seeds = 63 runs.
  - Experiments: {", ".join(TRAIN_EXPERIMENTS)}
- Noisy generalization runs: 2 experiments x 3 folds x 3 seeds = 18 runs.
  - Experiments: {", ".join(GENERALIZATION_EXPERIMENTS)}

## Files

- `coverage_summary.json`: completeness status (expected vs found).
- `train_manifest.csv`: one row per train run (accuracy/F1 + source mapping + sha256).
- `train_summary_by_experiment.csv`: per-experiment mean/std summary over 9 runs.
- `generalization_file_manifest.csv`: one row per generalization summary file.
- `generalization_long.csv`: expanded noisy metrics (group/snr rows across folds/seeds).
- `generalization_summary_by_group_snr.csv`: aggregated noisy summary by experiment/group/snr.
- `raw/`: copied source artifacts used to generate the tables.
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    print(f"[OK] output: {out_root}")
    print(
        f"[OK] coverage: train {len(train_map)}/{len(expected_train)}, "
        f"generalization {len(gen_map)}/{len(expected_gen)}"
    )


if __name__ == "__main__":
    main()
