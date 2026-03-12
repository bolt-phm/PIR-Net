#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi
OUT="${OUT:-$ROOT/results/merged}"

mkdir -p "$OUT"

for exp in 401 402 403 404; do
  if [ -d "$ROOT/$exp/logs" ]; then
    mkdir -p "$OUT/$exp"
    cp -r "$ROOT/$exp/logs/"* "$OUT/$exp/" 2>/dev/null || true
  fi
done

echo "[OK] merged logs -> $OUT"
