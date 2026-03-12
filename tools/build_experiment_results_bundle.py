#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PIRNET_IDS = ["022", "122", "202", "212", "220", "221", "222"]
BASELINE_IDS = ["301", "302", "303", "304", "305", "306", "307"]
ZENODO_IDS = ["401", "402", "403", "404"]

PIRNET_AGGREGATE_FILES = [
    "ablation_bar_chart.png",
    "confusion_matrix_comparison_fixed.png",
    "fig5_tsne.png",
    "fig6_attention_map.png",
    "5channel_representation.png",
    "resampling_comparison_detailed.png",
    "outlier_analysis_v2.png",
    "moxingliuchengtu.png",
    "moxingzonglan.png",
    "log_fanhau.log",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def rel_to_out(path: Path, out_root: Path) -> str:
    return str(path.relative_to(out_root)).replace("\\", "/")


def copy_with_index(
    src: Path,
    dst: Path,
    out_root: Path,
    index: list[dict],
    group: str,
    exp_id: str,
    artifact_type: str,
    notes: str = "",
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    index.append(
        {
            "group": group,
            "experiment_id": exp_id,
            "artifact_type": artifact_type,
            "file_path": rel_to_out(dst, out_root),
            "source_path": str(src).replace("\\", "/"),
            "status": "available",
            "size_bytes": dst.stat().st_size,
            "sha256": sha256_of(dst),
            "notes": notes,
        }
    )


def add_generated(index: list[dict], out_root: Path, path: Path, group: str, exp_id: str, artifact_type: str, notes: str) -> None:
    index.append(
        {
            "group": group,
            "experiment_id": exp_id,
            "artifact_type": artifact_type,
            "file_path": rel_to_out(path, out_root),
            "source_path": "",
            "status": "available",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "notes": notes,
        }
    )


def add_placeholder(index: list[dict], group: str, exp_id: str, artifact_type: str, expected_relative_path: str, notes: str) -> None:
    index.append(
        {
            "group": group,
            "experiment_id": exp_id,
            "artifact_type": artifact_type,
            "file_path": expected_relative_path,
            "source_path": "",
            "status": "expected_missing_in_snapshot",
            "size_bytes": 0,
            "sha256": "",
            "notes": notes,
        }
    )


def build_bundle(root: Path) -> None:
    out = root / "experiment_results"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source_fig = root / "paper_support" / "py" / "ppt_images"
    index: list[dict] = []

    # PIR-Net experiments
    for exp_id in PIRNET_IDS:
        exp_src = root / "experiments" / "pirnet_ablation" / exp_id
        exp_dst = out / "pirnet_ablation" / exp_id

        cfg_src = exp_src / "config.json"
        if cfg_src.exists():
            copy_with_index(
                cfg_src,
                exp_dst / "config_snapshot.json",
                out,
                index,
                "pirnet_ablation",
                exp_id,
                "config_snapshot",
                "Configuration snapshot for reproducible reruns.",
            )

        metrics_template_path = exp_dst / "metrics_summary_template.json"
        metrics_template = {
            "experiment_id": exp_id,
            "group": "pirnet_ablation",
            "status": "template",
            "metrics": {
                "best_val_accuracy": None,
                "test_accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "val_loss": None,
            },
            "notes": "Fill with finalized values exported from training/evaluation logs.",
        }
        write_json(metrics_template_path, metrics_template)
        add_generated(
            index,
            out,
            metrics_template_path,
            "pirnet_ablation",
            exp_id,
            "metrics_summary_template",
            "Template file for normalized experiment reporting.",
        )

        cm_src = source_fig / f"{exp_id}_confusion_matrix.png"
        if cm_src.exists():
            copy_with_index(
                cm_src,
                exp_dst / "artifacts" / "confusion_matrix.png",
                out,
                index,
                "pirnet_ablation",
                exp_id,
                "confusion_matrix",
                "Paper-support confusion matrix asset.",
            )
        else:
            add_placeholder(
                index,
                "pirnet_ablation",
                exp_id,
                "confusion_matrix",
                f"pirnet_ablation/{exp_id}/artifacts/confusion_matrix.png",
                "Expected after running generalization.py.",
            )

        add_placeholder(
            index,
            "pirnet_ablation",
            exp_id,
            "best_model_checkpoint",
            f"pirnet_ablation/{exp_id}/artifacts/best_model.pth",
            "Optional open artifact; include if model weights are intended for release.",
        )
        add_placeholder(
            index,
            "pirnet_ablation",
            exp_id,
            "train_log",
            f"pirnet_ablation/{exp_id}/artifacts/train.log",
            "Export from runtime logs if full traceability is required.",
        )

    # PIR-Net aggregate artifacts
    agg_dst = out / "pirnet_ablation" / "aggregate"
    for name in PIRNET_AGGREGATE_FILES:
        src = source_fig / name
        if src.exists():
            copy_with_index(
                src,
                agg_dst / name,
                out,
                index,
                "pirnet_ablation",
                "aggregate",
                "aggregate_figure",
                "Aggregate visualization generated during project analysis.",
            )

    # External baselines
    for exp_id in BASELINE_IDS:
        exp_src = root / "experiments" / "external_baselines" / exp_id
        exp_dst = out / "external_baselines" / exp_id

        cfg_src = exp_src / "config.json"
        if cfg_src.exists():
            copy_with_index(
                cfg_src,
                exp_dst / "config_snapshot.json",
                out,
                index,
                "external_baselines",
                exp_id,
                "config_snapshot",
                "Configuration snapshot for baseline reproducibility.",
            )

        metrics_template_path = exp_dst / "metrics_summary_template.json"
        metrics_template = {
            "experiment_id": exp_id,
            "group": "external_baselines",
            "status": "template",
            "metrics": {
                "best_val_accuracy": None,
                "test_accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "val_loss": None,
            },
            "notes": "Fill with baseline run metrics using the same reporting schema as PIR-Net.",
        }
        write_json(metrics_template_path, metrics_template)
        add_generated(
            index,
            out,
            metrics_template_path,
            "external_baselines",
            exp_id,
            "metrics_summary_template",
            "Template file for normalized baseline reporting.",
        )

        add_placeholder(
            index,
            "external_baselines",
            exp_id,
            "confusion_matrix",
            f"external_baselines/{exp_id}/artifacts/confusion_matrix.png",
            "Expected after running baseline generalization.py.",
        )
        add_placeholder(
            index,
            "external_baselines",
            exp_id,
            "best_model_checkpoint",
            f"external_baselines/{exp_id}/artifacts/best_model.pth",
            "Optional open artifact; include if weight release is planned.",
        )
        add_placeholder(
            index,
            "external_baselines",
            exp_id,
            "train_log",
            f"external_baselines/{exp_id}/artifacts/train.log",
            "Export from baseline training runs for full traceability.",
        )

    # Zenodo cross-condition experiments
    for exp_id in ZENODO_IDS:
        exp_src = root / "experiments" / "zenodo_generalization" / exp_id
        exp_dst = out / "zenodo_generalization" / exp_id

        cfg_src = exp_src / "config.json"
        if cfg_src.exists():
            copy_with_index(
                cfg_src,
                exp_dst / "config_snapshot.json",
                out,
                index,
                "zenodo_generalization",
                exp_id,
                "config_snapshot",
                "Configuration snapshot for cross-condition benchmark reproducibility.",
            )

        metrics_template_path = exp_dst / "metrics_summary_template.json"
        metrics_template = {
            "experiment_id": exp_id,
            "group": "zenodo_generalization",
            "status": "template",
            "metrics": {
                "best_val_accuracy": None,
                "test_accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "val_loss": None,
            },
            "notes": "Fill with Zenodo benchmark metrics using the same reporting schema.",
        }
        write_json(metrics_template_path, metrics_template)
        add_generated(
            index,
            out,
            metrics_template_path,
            "zenodo_generalization",
            exp_id,
            "metrics_summary_template",
            "Template file for standardized cross-condition benchmark reporting.",
        )

        add_placeholder(
            index,
            "zenodo_generalization",
            exp_id,
            "confusion_matrix",
            f"zenodo_generalization/{exp_id}/artifacts/confusion_matrix.png",
            "Expected after running Zenodo generalization evaluation.",
        )
        add_placeholder(
            index,
            "zenodo_generalization",
            exp_id,
            "best_model_checkpoint",
            f"zenodo_generalization/{exp_id}/artifacts/best_model.pth",
            "Optional open artifact; include if weight release is planned.",
        )
        add_placeholder(
            index,
            "zenodo_generalization",
            exp_id,
            "train_log",
            f"zenodo_generalization/{exp_id}/artifacts/train.log",
            "Export from Zenodo benchmark runs for full traceability.",
        )

    # Experiment design manifest
    plan = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "archive": "data.zip",
            "distribution": "Git LFS",
            "source": "self-collected FPGA-based impact-vibration acquisition",
        },
        "experiment_groups": {
            "pirnet_ablation": {
                "ids": PIRNET_IDS,
                "description": "PIR-Net ablation and main-model settings",
            },
            "external_baselines": {
                "ids": BASELINE_IDS,
                "description": "Unified external baselines without PIR-specific preprocessing",
            },
            "zenodo_generalization": {
                "ids": ZENODO_IDS,
                "description": "Cross-condition benchmark on external public dataset (Zenodo 15516419)",
            },
        },
        "artifact_policy": {
            "required_for_release": [
                "config snapshots",
                "metrics summary files",
                "confusion matrices",
                "aggregate figures",
            ],
            "optional_for_release": [
                "best checkpoints",
                "full training logs",
                "runtime traces",
            ],
        },
    }
    write_json(out / "experiment_plan.json", plan)

    # Save index files
    fieldnames = [
        "group",
        "experiment_id",
        "artifact_type",
        "file_path",
        "source_path",
        "status",
        "size_bytes",
        "sha256",
        "notes",
    ]
    with (out / "result_index.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index)

    write_json(out / "result_index.json", index)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(index),
        "available_records": sum(1 for x in index if x["status"] == "available"),
        "expected_missing_records": sum(1 for x in index if x["status"] == "expected_missing_in_snapshot"),
    }
    write_json(out / "summary.json", summary)

    readme = f"""# Experiment Results Bundle

This folder provides a structured, open-source-ready result bundle aligned with the project experiment design.

## Scope

- PIR-Net ablation experiments: {', '.join(PIRNET_IDS)}
- External baselines: {', '.join(BASELINE_IDS)}
- Zenodo cross-condition benchmarks: {', '.join(ZENODO_IDS)}

## Included Files

1. `experiment_plan.json`: experiment taxonomy and artifact policy.
2. `result_index.csv` / `result_index.json`: machine-readable artifact inventory with status and checksums.
3. `summary.json`: high-level counts of available and expected artifacts.
4. Per-experiment folders with:
   - `config_snapshot.json`
   - `metrics_summary_template.json`
   - copied available artifacts (for example confusion matrices where present)

## Current Snapshot Summary

- Total records: {summary['total_records']}
- Available records: {summary['available_records']}
- Expected-but-missing records: {summary['expected_missing_records']}

Expected-but-missing entries are placeholders for runtime outputs that are not currently tracked in this repository snapshot
(for example, full training logs and model checkpoints).

## Update Workflow

To regenerate this bundle from the latest repository contents:

```bash
python tools/build_experiment_results_bundle.py
```
"""
    (out / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    build_bundle(repo_root)
    print("Built experiment_results bundle.")
