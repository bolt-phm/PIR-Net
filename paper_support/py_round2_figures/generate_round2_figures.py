# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _repo_root() -> Path:
    # lw2/py_round2_figures/generate_round2_figures.py -> repo root
    return Path(__file__).resolve().parents[2]


def _load_results(merged_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(merged_csv)
    # Only keep the paper subset: 重做2/**/results/*
    df = df[df["source_path"].astype(str).str.contains("/results/")].copy()
    return df


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def fig_main_clean_ablation(df: pd.DataFrame, out_dir: Path) -> Path:
    clean = df[df["snr_db"] == "clean"].copy()
    main = clean[
        (clean["split_mode"] == "temporal")
        & (clean["alpha"] == 0.7)
        & (clean["compression_ratio"] == 150)
        & (clean["seed"] == 3407)
    ].copy()
    main = main.sort_values("experiment")

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = main["experiment"].astype(int).tolist()
    y = main["accuracy_pct"].astype(float).tolist()
    ax.bar([str(v).zfill(3) for v in x], y, color="#2E6F9E")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Main Ablation (Clean)  |  temporal, a=0.7, c=150, seed=3407")
    ax.grid(axis="y", alpha=0.25)
    for i, val in enumerate(y):
        ax.text(i, val + 0.6, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    out = out_dir / "fig_rev2_main_ablation_clean_temporal_a0p7_c150_seed3407.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_222_split_control_clean(df: pd.DataFrame, out_dir: Path) -> Path:
    clean = df[df["snr_db"] == "clean"].copy()
    sub = clean[
        (clean["experiment"] == 222)
        & (clean["alpha"] == 0.7)
        & (clean["compression_ratio"] == 150)
        & (clean["seed"] == 3407)
    ].copy()
    sub = sub.sort_values("split_mode")

    # Majority-class baseline on file split (from report: SevereLoose support 5832 / 31104)
    majority_baseline = 5832.0 / 31104.0 * 100.0

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = []
    vals = []
    for _, r in sub.iterrows():
        labels.append(f"{r['split_mode']}\n(n={int(r['n_test_samples'])})")
        vals.append(float(r["accuracy_pct"]))

    ax.bar(labels, vals, color=["#8A2BE2" if "file" in l else "#2E6F9E" for l in labels])
    ax.axhline(majority_baseline, color="#444444", linestyle="--", linewidth=1.2, label="File-split majority baseline")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Split Strategy Control (Clean)  |  Exp 222, a=0.7, c=150, seed=3407")
    ax.grid(axis="y", alpha=0.25)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.6, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = out_dir / "fig_rev2_split_control_222_clean_seed3407.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_222_alpha_sensitivity_clean(df: pd.DataFrame, out_dir: Path) -> Path:
    clean = df[df["snr_db"] == "clean"].copy()
    sub = clean[
        (clean["experiment"] == 222)
        & (clean["split_mode"] == "temporal")
        & (clean["compression_ratio"] == 150)
        & (clean["seed"] == 3407)
    ].copy()
    sub = sub.sort_values("alpha")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = [f"{a:.1f}" for a in sub["alpha"].astype(float).tolist()]
    y = sub["accuracy_pct"].astype(float).tolist()
    ax.plot(x, y, marker="o", linewidth=2.0, color="#2E6F9E")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Alpha (peak/mean mixing)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Alpha Sensitivity (Clean)  |  Exp 222 temporal, c=150, seed=3407")
    ax.grid(alpha=0.25)
    for i, val in enumerate(y):
        ax.text(i, val + 0.6, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    out = out_dir / "fig_rev2_alpha_sensitivity_222_clean_temporal_c150_seed3407.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_222_compression_sensitivity_clean(df: pd.DataFrame, out_dir: Path) -> Path:
    clean = df[df["snr_db"] == "clean"].copy()
    sub = clean[
        (clean["experiment"] == 222)
        & (clean["split_mode"] == "temporal")
        & (clean["alpha"] == 0.7)
        & (clean["seed"] == 3407)
    ].copy()
    sub = sub.sort_values("compression_ratio")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = [str(int(c)) for c in sub["compression_ratio"].astype(int).tolist()]
    y = sub["accuracy_pct"].astype(float).tolist()
    ax.plot(x, y, marker="o", linewidth=2.0, color="#2E6F9E")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Compression ratio (raw_len / target_len)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Compression Sensitivity (Clean)  |  Exp 222 temporal, a=0.7, seed=3407")
    ax.grid(alpha=0.25)
    for i, val in enumerate(y):
        ax.text(i, val + 0.6, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    out = out_dir / "fig_rev2_compression_sensitivity_222_clean_temporal_a0p7_seed3407.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_222_robustness_curve(df: pd.DataFrame, out_dir: Path) -> Path:
    sub = df[
        (df["experiment"] == 222)
        & (df["split_mode"] == "temporal")
        & (df["alpha"] == 0.7)
        & (df["compression_ratio"] == 150)
        & (df["seed"] == 3407)
    ].copy()

    def _snr_to_x(v) -> float | None:
        # merged_all_runs.csv stores snr_db as strings like '20.0', '-5.0', or 'clean'
        s = str(v).strip().lower()
        if s == "clean":
            return 30.0  # plot clean at +30 dB for compactness
        try:
            return float(s)
        except Exception:
            return None

    sub["snr_x"] = sub["snr_db"].apply(_snr_to_x)
    sub = sub.dropna(subset=["snr_x"]).sort_values(["seed", "snr_x"])

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(
        sub["snr_x"].astype(float).tolist(),
        sub["accuracy_pct"].astype(float).tolist(),
        marker="o",
        linewidth=2.0,
        color="#2E6F9E",
        label="Exp 222 (seed=3407)",
    )
    ax.set_xlim(-6, 31)
    ax.set_ylim(0, 100)
    ax.set_xlabel("SNR (dB)   (clean plotted at +30)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness vs. SNR  |  Exp 222 temporal, a=0.7, c=150, seed=3407")
    ax.set_xticks([-5, 0, 5, 10, 20, 30])
    ax.set_xticklabels(["-5", "0", "5", "10", "20", "clean"])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out = out_dir / "fig_rev2_robustness_curve_222_temporal_a0p7_c150_seed2026_3407.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--merged_csv",
        type=str,
        default=str(_repo_root() / "重做2" / "analysis" / "merged_all_runs.csv"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_repo_root() / "lw2" / "figures"),
    )
    args = ap.parse_args()

    merged_csv = Path(args.merged_csv)
    out_dir = Path(args.out_dir)
    _ensure_out(out_dir)
    _style()

    df = _load_results(merged_csv)

    outputs = [
        fig_main_clean_ablation(df, out_dir),
        fig_222_split_control_clean(df, out_dir),
        fig_222_alpha_sensitivity_clean(df, out_dir),
        fig_222_compression_sensitivity_clean(df, out_dir),
        fig_222_robustness_curve(df, out_dir),
    ]
    for p in outputs:
        print(f"[OK] wrote: {p}")


if __name__ == "__main__":
    main()
