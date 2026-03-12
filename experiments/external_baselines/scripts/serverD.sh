#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/up/baselines}"
cd "$ROOT"

python run_baselines.py --help >/dev/null
# Throughput tuning for dataloading and CPU thread affinity
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$ROOT/results"

# Server D: 307 regular SOTA-style baseline
python run_baselines.py --experiments 307

mkdir -p "$ROOT/results/serverD"
cp -r "$ROOT/307/logs" "$ROOT/results/serverD/307_logs" 2>/dev/null || true
