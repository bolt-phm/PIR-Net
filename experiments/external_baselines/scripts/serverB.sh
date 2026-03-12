#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/up/baselines}"
cd "$ROOT"

python run_baselines.py --help >/dev/null
mkdir -p "$ROOT/results"

# Server B: 1D signal baselines (part 2)
python run_baselines.py --experiments 303 306

mkdir -p "$ROOT/results/serverB"
cp -r "$ROOT/303/logs" "$ROOT/results/serverB/303_logs" 2>/dev/null || true
cp -r "$ROOT/306/logs" "$ROOT/results/serverB/306_logs" 2>/dev/null || true
