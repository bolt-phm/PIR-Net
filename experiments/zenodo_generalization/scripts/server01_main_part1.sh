#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"
mkdir -p "$ROOT/results"

# Supplement-only (R3): external baseline 401 clean CV.
python run_cv_experiments.py \
  --experiments 401 403 \
  --folds fold_d15_cfg22 \
  --seeds 3407 2026 4096 \
  --eval_workers 8 \
  --num_workers_train_override 20 \
  --batch_size_train_override 640 \
  --batch_size_eval_override 1024 \
  --prefetch_factor_override 8 \
  --enable_ram_cache \
  --patience_main 25 \
  --patience_other 10 \
  --skip_existing

python run_cv_experiments.py \
  --experiments 501 503 \
  --folds fold_d15_cfg22 \
  --seeds 3407 2026 4096 \
  --with_generalization \
  --generalization_experiments 501 \
  --snrs clean 10 5 0 \
  --snr_repeats 3 \
  --eval_workers 8 \
  --num_workers_train_override 14 \
  --batch_size_train_override 192 \
  --batch_size_eval_override 256 \
  --prefetch_factor_override 6 \
  --enable_ram_cache \
  --patience_main 25 \
  --patience_other 10 \
  --skip_existing
