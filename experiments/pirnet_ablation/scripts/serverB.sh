#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/up}"
cd "$ROOT"

python run_paper_pipeline.py --help | grep -q -- '--skip_train_if_exists' || { echo "ERROR: run_paper_pipeline.py is outdated (missing --skip_train_if_exists). Update it and retry."; exit 1; }
mkdir -p "$ROOT/results"

# Main ablation (temporal, seed=3407): 202 + 212
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 202 212 --split_modes temporal --alphas 0.7 --compression_ratios 150 --seeds 3407 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverB_main_seed3407_202_212"

# Alpha sensitivity (temporal, seed=3407): alpha=0.9 only
python run_paper_pipeline.py --root "$ROOT" --sync_sources --train --skip_train_if_exists \
  --experiments 222 --split_modes temporal --alphas 0.9 --compression_ratios 150 --seeds 3407 \
  --snrs clean 20 10 5 0 -5 --out_dir "$ROOT/results/serverB_222_alpha0p9_seed3407"
