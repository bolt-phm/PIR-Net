#!/usr/bin/env bash
set -euo pipefail

# ROOT should point to the baselines folder after upload.
ROOT="${ROOT:-$HOME/autodl-tmp/up/baselines}"
cd "$ROOT"

python run_baselines.py --help >/dev/null
mkdir -p "$ROOT/results"

# Server A: 1D signal baselines (part 1)
python run_baselines.py --experiments 301 302

# Archive logs for easier collection
mkdir -p "$ROOT/results/serverA"
cp -r "$ROOT/301/logs" "$ROOT/results/serverA/301_logs" 2>/dev/null || true
cp -r "$ROOT/302/logs" "$ROOT/results/serverA/302_logs" 2>/dev/null || true
