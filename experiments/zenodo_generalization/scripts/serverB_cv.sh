#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/results/serverB_cv"
python version_fingerprint.py --project_root "$ROOT" --exp 402 --out "$ROOT/results/serverB_cv/version_402.json"
python run_cv_experiments.py --experiments 402 --with_generalization
cp -r "$ROOT/402/logs" "$ROOT/results/serverB_cv/402_logs" 2>/dev/null || true
