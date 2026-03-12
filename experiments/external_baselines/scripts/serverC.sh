#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/up/baselines}"
cd "$ROOT"

python run_baselines.py --help >/dev/null
mkdir -p "$ROOT/results"

# Server C: CNN-based spectrogram baseline
python run_baselines.py --experiments 304

mkdir -p "$ROOT/results/serverC"
cp -r "$ROOT/304/logs" "$ROOT/results/serverC/304_logs" 2>/dev/null || true
