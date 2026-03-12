#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi
cd "$ROOT"

echo "[INFO] root: $ROOT"
python version_fingerprint.py --project_root "$ROOT" --exp 401
python inspect_dataset.py --exp_dir ./401
echo "[OK] preflight finished"
