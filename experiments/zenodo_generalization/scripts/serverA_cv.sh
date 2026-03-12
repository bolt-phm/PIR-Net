#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/results/serverA_cv"
python version_fingerprint.py --project_root "$ROOT" --exp 401 --out "$ROOT/results/serverA_cv/version_401.json"
python run_cv_experiments.py --experiments 401 --with_generalization
cp -r "$ROOT/401/logs" "$ROOT/results/serverA_cv/401_logs" 2>/dev/null || true
