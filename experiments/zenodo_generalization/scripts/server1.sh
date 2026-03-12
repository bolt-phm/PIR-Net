#!/usr/bin/env bash
set -euo pipefail

# ROOT should point to this project directory on the server.
ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$ROOT/results/server1"
python version_fingerprint.py --project_root "$ROOT" --exp 401 --out "$ROOT/results/server1/version_401.json"
python version_fingerprint.py --project_root "$ROOT" --exp 402 --out "$ROOT/results/server1/version_402.json"

python run_experiments.py --experiments 401 402 --with_generalization

cp -r "$ROOT/401/logs" "$ROOT/results/server1/401_logs" 2>/dev/null || true
cp -r "$ROOT/402/logs" "$ROOT/results/server1/402_logs" 2>/dev/null || true
