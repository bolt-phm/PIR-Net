#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/up/baselines}"
cd "$ROOT"

python run_baselines.py --help >/dev/null
mkdir -p "$ROOT/results"

# Server D: SOTA-like spectrogram baseline
python run_baselines.py --experiments 305

mkdir -p "$ROOT/results/serverD"
cp -r "$ROOT/305/logs" "$ROOT/results/serverD/305_logs" 2>/dev/null || true
