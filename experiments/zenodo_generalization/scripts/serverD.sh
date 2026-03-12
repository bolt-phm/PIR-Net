#!/usr/bin/env bash
set -euo pipefail

# ROOT should point to this project directory on the server.
ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/results/serverD"
python version_fingerprint.py --project_root "$ROOT" --exp 404 --out "$ROOT/results/serverD/version_404.json"
python run_experiments.py --experiments 404 --with_generalization

cp -r "$ROOT/404/logs" "$ROOT/results/serverD/404_logs" 2>/dev/null || true
