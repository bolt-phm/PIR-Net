#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

count_train_done() {
  local exp="$1"
  find "$ROOT/$exp" -path "*/logs/cv_*/*/metrics_clean.json" 2>/dev/null | wc -l | tr -d ' '
}

count_gen_done() {
  local exp="$1"
  find "$ROOT/$exp" -path "*/logs/cv_*/*/generalization_summary.csv" 2>/dev/null | wc -l | tr -d ' '
}

expected_per_exp=9

t401=$(count_train_done 401)
t402=$(count_train_done 402)
t403=$(count_train_done 403)
t404=$(count_train_done 404)
t501=$(count_train_done 501)
t502=$(count_train_done 502)
t503=$(count_train_done 503)

g404=$(count_gen_done 404)
g501=$(count_gen_done 501)

train_total=$((t401 + t402 + t403 + t404 + t501 + t502 + t503))
gen_total=$((g404 + g501))

echo "train 401: ${t401}/${expected_per_exp}"
echo "train 402: ${t402}/${expected_per_exp}"
echo "train 403: ${t403}/${expected_per_exp}"
echo "train 404: ${t404}/${expected_per_exp}"
echo "train 501 (PIR): ${t501}/${expected_per_exp}"
echo "train 502 (AVG ablation): ${t502}/${expected_per_exp}"
echo "train 503 (DECIMATE ablation): ${t503}/${expected_per_exp}"
echo "generalization 404: ${g404}/${expected_per_exp}"
echo "generalization 501: ${g501}/${expected_per_exp}"
echo "TOTAL train: ${train_total}/63 | TOTAL generalization: ${gen_total}/18"
