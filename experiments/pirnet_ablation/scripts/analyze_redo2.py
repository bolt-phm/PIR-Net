# -*- coding: utf-8 -*-
"""
Analyze experimental results under 重做2/.

Outputs:
  - 重做2/analysis/merged_all_runs.csv
  - 重做2/analysis/summary.md
  - 重做2/analysis/dup_conflicts.csv (if any)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPECTED_SNRS = ["clean", 20.0, 10.0, 5.0, 0.0, -5.0]


def _to_float_snr(x):
    if pd.isna(x):
        return "clean"
    if isinstance(x, str) and x.strip().lower() == "clean":
        return "clean"
    return float(x)


def _norm_snr_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["snr_db"] = out["snr_db"].map(_to_float_snr)
    return out


def load_all_runs(csv_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["source_path"] = str(p.as_posix())
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = _norm_snr_col(merged)
    return merged


def find_all_runs(redo_dir: Path) -> list[Path]:
    return sorted(redo_dir.rglob("all_runs.csv"))


def dedupe_with_conflict_check(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Deduplicate by the true run identity. If multiple sources disagree on accuracy for the same key, flag it.
    """
    key_cols = ["experiment", "split_mode", "alpha", "compression_ratio", "seed", "snr_db", "n_test_samples"]
    df2 = df.copy()

    # Detect conflicts: same key but different accuracy.
    g = df2.groupby(key_cols, dropna=False)
    conflicts = g["accuracy_pct"].nunique().reset_index(name="n_unique_acc")
    conflicts = conflicts[conflicts["n_unique_acc"] > 1]
    if len(conflicts) > 0:
        conflict_keys = conflicts[key_cols]
        conflict_rows = df2.merge(conflict_keys, on=key_cols, how="inner").sort_values(key_cols)
        out_dir.mkdir(parents=True, exist_ok=True)
        conflict_rows.to_csv(out_dir / "dup_conflicts.csv", index=False)

    # Keep the latest row by timestamp if duplicated.
    if "timestamp" in df2.columns:
        df2["_ts"] = pd.to_datetime(df2["timestamp"], errors="coerce")
        df2 = df2.sort_values("_ts")
    else:
        df2["_ts"] = pd.NaT

    df2 = df2.drop_duplicates(subset=key_cols, keep="last").drop(columns=["_ts"], errors="ignore")
    return df2


@dataclass(frozen=True)
class RunKey:
    experiment: str
    split_mode: str
    alpha: float
    compression_ratio: int
    seed: int


def summarize_run_coverage(df: pd.DataFrame) -> pd.DataFrame:
    run_cols = ["experiment", "split_mode", "alpha", "compression_ratio", "seed"]
    cov = (
        df.groupby(run_cols, dropna=False)["snr_db"]
        .apply(lambda s: sorted(set(s.tolist()), key=lambda x: (x != "clean", float(x) if x != "clean" else 0.0)))
        .reset_index(name="snrs_present")
    )
    cov["has_all_expected_snrs"] = cov["snrs_present"].apply(
        lambda xs: set(xs) == set(EXPECTED_SNRS)
    )
    return cov


def _clean_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["snr_db"] == "clean"].copy()


def _pivot_clean(df: pd.DataFrame) -> pd.DataFrame:
    c = _clean_only(df)
    cols = ["experiment", "split_mode", "alpha", "compression_ratio", "seed", "accuracy_pct", "n_test_samples"]
    return c[cols].sort_values(["experiment", "split_mode", "alpha", "compression_ratio", "seed"])


def _compare_222_file_vs_temporal(clean_df: pd.DataFrame) -> pd.DataFrame:
    d = clean_df[clean_df["experiment"] == 222].copy()
    if d.empty:
        return pd.DataFrame()
    keep = ["split_mode", "alpha", "compression_ratio", "seed", "accuracy_pct", "n_test_samples"]
    d = d[keep].sort_values(["split_mode", "alpha", "compression_ratio", "seed"])
    return d


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "merged_all_runs.csv"
    df.to_csv(merged_path, index=False)

    def _render_one(title: str, dframe: pd.DataFrame, filename: str) -> None:
        cov = summarize_run_coverage(dframe)
        clean = _clean_only(dframe)

        # Core focused comparisons (for paper response)
        main_clean = clean[(clean["split_mode"] == "temporal") & (clean["alpha"] == 0.7) & (clean["compression_ratio"] == 150)]
        main_clean = main_clean.sort_values(["experiment", "seed"])

        alpha_sens = clean[(clean["experiment"] == 222) & (clean["split_mode"] == "temporal") & (clean["compression_ratio"] == 150)]
        alpha_sens = alpha_sens.sort_values(["alpha", "seed"])

        comp_sens = clean[(clean["experiment"] == 222) & (clean["split_mode"] == "temporal") & (clean["alpha"] == 0.7)]
        comp_sens = comp_sens.sort_values(["compression_ratio", "seed"])

        split_cmp_222 = _compare_222_file_vs_temporal(clean)

        # Robustness slice for 222 main settings (temporal, a=0.7,c=150)
        robust_222 = dframe[(dframe["experiment"] == 222) & (dframe["split_mode"] == "temporal") & (dframe["alpha"] == 0.7) & (dframe["compression_ratio"] == 150)]
        robust_222 = robust_222.sort_values(["seed", "snr_db"])

        # Render markdown (keep it compact but informative)
        md = []
        md.append(f"# {title}\n")
        md.append(f"- Total rows (after dedupe): {len(dframe)}\n")
        md.append(f"- Unique runs (exp/split/alpha/comp/seed): {len(cov)}\n")
        md.append(f"- CSV merged: {merged_path.as_posix()}\n")
        md.append("\n## 运行覆盖\n")
        md.append(cov.sort_values(["experiment", "split_mode", "alpha", "compression_ratio", "seed"]).to_markdown(index=False))

        md.append("\n\n## 主设置 (temporal, a=0.7, c=150) clean\n")
        if len(main_clean) == 0:
            md.append("_No rows._\n")
        else:
            md.append(main_clean[["experiment", "seed", "accuracy_pct", "n_test_samples"]].to_markdown(index=False))

        md.append("\n\n## 222 随机种子对比 (temporal, a=0.7, c=150) clean\n")
        tmp = clean[(clean["experiment"] == 222) & (clean["split_mode"] == "temporal") & (clean["alpha"] == 0.7) & (clean["compression_ratio"] == 150)]
        if len(tmp) == 0:
            md.append("_No rows._\n")
        else:
            md.append(tmp[["seed", "accuracy_pct", "n_test_samples"]].sort_values(["seed"]).to_markdown(index=False))

        md.append("\n\n## 222 file 对照 vs temporal (clean)\n")
        if len(split_cmp_222) == 0:
            md.append("_No rows._\n")
        else:
            md.append(split_cmp_222.to_markdown(index=False))

        md.append("\n\n## 222 alpha 敏感性 (temporal, c=150, seed=3407/2026)\n")
        if len(alpha_sens) == 0:
            md.append("_No rows._\n")
        else:
            md.append(alpha_sens[["alpha", "seed", "accuracy_pct", "n_test_samples"]].to_markdown(index=False))

        md.append("\n\n## 222 压缩比敏感性 (temporal, a=0.7)\n")
        if len(comp_sens) == 0:
            md.append("_No rows._\n")
        else:
            md.append(comp_sens[["compression_ratio", "seed", "accuracy_pct", "n_test_samples"]].to_markdown(index=False))

        md.append("\n\n## 222 鲁棒性曲线 (temporal, a=0.7, c=150)\n")
        if len(robust_222) == 0:
            md.append("_No rows._\n")
        else:
            md.append(robust_222[["seed", "snr_db", "accuracy_pct", "n_test_samples"]].to_markdown(index=False))

        summary_path = out_dir / filename
        summary_path.write_text("\n".join(md), encoding="utf-8")

    _render_one("重做2 实验结果汇总（全量，含 paper_outputs_*）", df, "summary_all.md")

    results_df = df[df["source_path"].str.contains("/results/")].copy()
    _render_one("重做2 实验结果汇总（论文子集，仅 results/*）", results_df, "summary_results.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo_dir", type=str, default="重做2")
    ap.add_argument("--out_dir", type=str, default=os.path.join("重做2", "analysis"))
    args = ap.parse_args()

    redo_dir = Path(args.redo_dir)
    out_dir = Path(args.out_dir)

    csvs = find_all_runs(redo_dir)
    if not csvs:
        raise SystemExit(f"No all_runs.csv found under: {redo_dir}")

    df = load_all_runs(csvs)
    df = dedupe_with_conflict_check(df, out_dir=out_dir)

    write_summary(df, out_dir=out_dir)
    print(f"[OK] wrote: {(out_dir / 'summary_all.md').as_posix()}")
    print(f"[OK] wrote: {(out_dir / 'summary_results.md').as_posix()}")


if __name__ == "__main__":
    main()
