#!/usr/bin/env python3
import csv
import os
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[1]))
out_dir = ROOT / "results"
out_dir.mkdir(parents=True, exist_ok=True)

all_csv = sorted(ROOT.glob("*/logs/all_runs.csv"))
out_path = out_dir / "combined_all_runs.csv"

rows = []
fieldnames = None
for p in all_csv:
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        for r in reader:
            r["source_csv"] = str(p)
            rows.append(r)

if not rows:
    print("No all_runs.csv found under baselines/*/logs")
    raise SystemExit(0)

if "source_csv" not in fieldnames:
    fieldnames = list(fieldnames) + ["source_csv"]

with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows -> {out_path}")
