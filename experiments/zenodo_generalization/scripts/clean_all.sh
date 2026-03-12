#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/autodl-tmp/zenodo15516419_damage_generalization}"
if [ ! -d "$ROOT" ]; then
  ROOT="$HOME/autodl-tmp/up/zenodo15516419_damage_generalization"
fi

cd "$ROOT"
for exp in 401 402 403 404; do
  rm -rf "$ROOT/$exp/logs" "$ROOT/$exp/checkpoints" "$ROOT/$exp/.cv_configs"
done

echo "[OK] cleaned logs/checkpoints for 401-404 under $ROOT"
