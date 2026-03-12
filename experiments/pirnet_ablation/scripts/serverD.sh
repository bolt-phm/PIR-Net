#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/up}"
cd "$ROOT"

python run_paper_pipeline.py --help | grep -q -- '--skip_train_if_exists' || { echo "ERROR: run_paper_pipeline.py is outdated (missing --skip_train_if_exists). Update it and retry."; exit 1; }
mkdir -p "$ROOT/results"

# Main setting (temporal, seed=3407): 222
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 222 --split_modes temporal --alphas 0.7 --compression_ratios 150 --seeds 3407 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverD_222_main_seed3407"

# Random seed robustness (temporal, seed=2026): 222 (no seed=256)
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 222 --split_modes temporal --alphas 0.7 --compression_ratios 150 --seeds 2026 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverD_222_main_seed2026"

# Compression sensitivity (temporal, seed=3407): comp=100
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 222 --split_modes temporal --alphas 0.7 --compression_ratios 100 --seeds 3407 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverD_222_comp100_seed3407"

# Compression sensitivity (temporal, seed=3407): comp=200
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 222 --split_modes temporal --alphas 0.7 --compression_ratios 200 --seeds 3407 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverD_222_comp200_seed3407"
