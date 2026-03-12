#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/results/serverC_cv"
python version_fingerprint.py --project_root "$ROOT" --exp 403 --out "$ROOT/results/serverC_cv/version_403.json"
python run_cv_experiments.py --experiments 403 --with_generalization
cp -r "$ROOT/403/logs" "$ROOT/results/serverC_cv/403_logs" 2>/dev/null || true
