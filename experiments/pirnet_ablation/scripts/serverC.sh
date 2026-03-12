#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/up}"
cd "$ROOT"

python run_paper_pipeline.py --help | grep -q -- '--skip_train_if_exists' || { echo "ERROR: run_paper_pipeline.py is outdated (missing --skip_train_if_exists). Update it and retry."; exit 1; }
mkdir -p "$ROOT/results"

# Main ablation (temporal, seed=3407): 220 + 221
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 220 221 --split_modes temporal --alphas 0.7 --compression_ratios 150 --seeds 3407 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverC_main_seed3407_220_221"

# Leakage-avoidance control (file split, seed=3407): 222 (only 1 run as control)
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 222 --split_modes file --alphas 0.7 --compression_ratios 150 --seeds 3407 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverC_222_file_seed3407"
