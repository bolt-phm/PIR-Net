#!/usr/bin/env python3
"""Lightweight smoke checks for PIR-Net repository health."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import tempfile
from pathlib import Path

from tools.update_data_dir import update_config


IMPORT_MODULES = [
    "dataset",
    "model",
    "train",
    "generalization",
    "inference_engine",
]


def run_import_smoke() -> None:
    failures: list[tuple[str, str]] = []
    for name in IMPORT_MODULES:
        try:
            importlib.import_module(name)
            print(f"[OK] import {name}")
        except Exception as exc:  # pragma: no cover - runtime validation path
            failures.append((name, str(exc)))
            print(f"[FAIL] import {name}: {exc}")

    if failures:
        detail = "\n".join(f"- {n}: {m}" for n, m in failures)
        raise RuntimeError(f"Import smoke test failed:\n{detail}")


def run_config_dry_run(exp_dir: Path) -> None:
    cfg_path = exp_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    required_top = ("data", "train")
    for key in required_top:
        if key not in cfg:
            raise KeyError(f"Missing required key in config: {key}")

    data_keys = ("data_dir", "num_classes")
    train_keys = ("model_dir", "best_model_name")
    for key in data_keys:
        if key not in cfg["data"]:
            raise KeyError(f"Missing data.{key} in config")
    for key in train_keys:
        if key not in cfg["train"]:
            raise KeyError(f"Missing train.{key} in config")

    with tempfile.TemporaryDirectory(prefix="pirnet_smoke_") as tmp:
        tmp_cfg = Path(tmp) / "config.json"
        shutil.copy2(cfg_path, tmp_cfg)

        changed = update_config(
            path=tmp_cfg,
            data_dir="/tmp/pirnet_data",
            generalization_dir="/tmp/pirnet_generalization",
        )
        if not changed:
            raise RuntimeError("Dry-run update expected to change config but no change occurred")

        with tmp_cfg.open("r", encoding="utf-8") as f:
            updated = json.load(f)

        if updated["data"].get("data_dir") != "/tmp/pirnet_data":
            raise RuntimeError("Dry-run update failed to apply data_dir")
        if updated["data"].get("generalization_dir") != "/tmp/pirnet_generalization":
            raise RuntimeError("Dry-run update failed to apply generalization_dir")

    print(f"[OK] config dry-run ({cfg_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight repository smoke checks")
    parser.add_argument("--mode", choices=["all", "import", "config"], default="all")
    parser.add_argument("--exp_dir", type=Path, default=Path("experiments/pirnet_ablation/222"))
    args = parser.parse_args()

    if args.mode in ("all", "import"):
        run_import_smoke()
    if args.mode in ("all", "config"):
        run_config_dry_run(args.exp_dir)

    print("[OK] smoke checks completed")


if __name__ == "__main__":
    main()