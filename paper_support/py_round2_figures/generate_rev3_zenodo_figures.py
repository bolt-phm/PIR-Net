# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_META = {
    "401": "WDCNN",
    "402": "InceptionTime",
    "403": "ResNet1D",
    "404": "CNN-BiLSTM-Attn",
}

LOG_ROOTS = {
    "401": ("A", "401"),
    "402": ("B", "402"),
    "403": ("C", "403"),
    "404": ("D", "404"),
}

FOLD_ORDER = ["d15_cfg22", "d16_cfg21", "d17_cfg21"]
SNR_ORDER = ["clean", "10.0", "5.0", "0.0"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_float_snr(v: str) -> float:
    if v == "clean":
        return 30.0
    return float(v)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _parse_fold(exp_name: str) -> str:
    # e.g., 401_fold_d15_cfg22_s3407
    m = re.search(r"_fold_(d\d+_cfg\d+)_", exp_name)
    if m:
        return m.group(1)
    return "unknown"


def load_round3_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict] = []
    noise_rows: list[dict] = []

    for exp_id, (server_tag, exp_dir) in LOG_ROOTS.items():
        log_dir = repo_root / "zenodo" / server_tag / exp_dir / "logs"
        cv_summary_path = log_dir / "cv_summary.csv"
        if not cv_summary_path.exists():
            raise FileNotFoundError(f"Missing cv summary: {cv_summary_path}")

        cv_df = pd.read_csv(cv_summary_path)
        if cv_df.empty:
            continue

        for _, row in cv_df.iterrows():
            exp_name = str(row["experiment"])
            fold_name = _parse_fold(exp_name)
            seed = int(row["seed"])
            fold_rows.append(
                {
                    "exp_id": exp_id,
                    "model": EXP_META[exp_id],
                    "fold": fold_name,
                    "seed": seed,
                    "accuracy_pct": float(row["accuracy_pct"]),
                    "macro_f1": float(row["macro_f1"]),
                    "weighted_f1": float(row["weighted_f1"]),
                }
            )

            gen_path = log_dir / f"cv_fold_{fold_name}" / f"seed_{seed}" / "generalization_runs.csv"
            if not gen_path.exists():
                continue
            gen_df = pd.read_csv(gen_path)
            gen_df = gen_df[gen_df["group"] == "fold_test"].copy()
            if gen_df.empty:
                continue
            gen_df["snr_db"] = gen_df["snr_db"].astype(str).str.strip().str.lower()
            for snr in SNR_ORDER:
                sub = gen_df[gen_df["snr_db"] == snr]
                if sub.empty:
                    continue
                noise_rows.append(
                    {
                        "exp_id": exp_id,
                        "model": EXP_META[exp_id],
                        "fold": fold_name,
                        "snr_db": snr,
                        "macro_f1": float(sub["macro_f1"].mean()),
                        "accuracy_pct": float(sub["accuracy_pct"].mean()),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    noise_df = pd.DataFrame(noise_rows)
    if fold_df.empty or noise_df.empty:
        raise RuntimeError("Round-3 fold/noise data is empty; cannot generate figures.")
    return fold_df, noise_df


def save_summary_csvs(fold_df: pd.DataFrame, noise_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_summary = (
        fold_df.groupby(["exp_id", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy_pct", "mean"),
            accuracy_std=("accuracy_pct", lambda x: float(np.std(np.asarray(x, dtype=float), ddof=0))),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda x: float(np.std(np.asarray(x, dtype=float), ddof=0))),
        )
        .sort_values("exp_id")
    )
    clean_summary.to_csv(out_dir / "rev3_zenodo_cv_clean_summary.csv", index=False)

    fold_detail = fold_df.sort_values(["exp_id", "fold"])
    fold_detail.to_csv(out_dir / "rev3_zenodo_cv_fold_details.csv", index=False)

    noise_summary = (
        noise_df.groupby(["exp_id", "model", "snr_db"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda x: float(np.std(np.asarray(x, dtype=float), ddof=0))),
        )
        .sort_values(["exp_id", "snr_db"])
    )
    noise_summary.to_csv(out_dir / "rev3_zenodo_noise_summary.csv", index=False)


def fig_cv_clean_macrof1(fold_df: pd.DataFrame, out_dir: Path) -> Path:
    summary = (
        fold_df.groupby(["exp_id", "model"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda x: float(np.std(np.asarray(x, dtype=float), ddof=0))),
        )
        .sort_values("exp_id")
    )

    labels = [f"{r.exp_id}\n{r.model}" for r in summary.itertuples()]
    means = summary["macro_f1_mean"].to_numpy()
    stds = summary["macro_f1_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color="#2A6F97", edgecolor="black", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Zenodo external generalization (clean, 3-fold CV, fold_test)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + 0.03,
            f"{m:.3f}\u00b1{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    out_path = out_dir / "fig_rev3_zenodo_cv_clean_macrof1.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_fold_heatmap(fold_df: pd.DataFrame, out_dir: Path) -> Path:
    pivot = (
        fold_df.pivot_table(index=["exp_id", "model"], columns="fold", values="macro_f1", aggfunc="mean")
        .reindex(columns=FOLD_ORDER)
        .sort_index()
    )
    mat = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(FOLD_ORDER)))
    ax.set_xticklabels(FOLD_ORDER)
    ylabels = [f"{idx[0]} {idx[1]}" for idx in pivot.index]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Fold-wise macro-F1 (clean, fold_test)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Macro-F1")
    fig.tight_layout()
    out_path = out_dir / "fig_rev3_zenodo_fold_macrof1_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig_noise_curve(noise_df: pd.DataFrame, out_dir: Path) -> Path:
    summary = (
        noise_df.groupby(["exp_id", "model", "snr_db"], as_index=False)
        .agg(macro_f1_mean=("macro_f1", "mean"))
        .copy()
    )
    summary = summary[summary["snr_db"].isin(SNR_ORDER)].copy()
    summary["snr_x"] = summary["snr_db"].map(_to_float_snr)

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    palette = {
        "401": "#1F77B4",
        "402": "#2CA02C",
        "403": "#FF7F0E",
        "404": "#D62728",
    }
    for exp_id in ["401", "402", "403", "404"]:
        sub = summary[summary["exp_id"] == exp_id].sort_values("snr_x")
        ax.plot(
            sub["snr_x"].to_numpy(),
            sub["macro_f1_mean"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=palette[exp_id],
            label=f"{exp_id} ({EXP_META[exp_id]})",
        )
    ax.set_xticks([30.0, 10.0, 5.0, 0.0])
    ax.set_xticklabels(["clean", "10", "5", "0"])
    ax.set_ylim(0.25, 1.0)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Zenodo noise robustness (mean over 3 folds, fold_test)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path = out_dir / "fig_rev3_zenodo_noise_robustness.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    repo_root = _repo_root()
    out_dir = repo_root / "lw2" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    fold_df, noise_df = load_round3_data(repo_root)
    save_summary_csvs(fold_df, noise_df, out_dir)

    outputs = [
        fig_cv_clean_macrof1(fold_df, out_dir),
        fig_fold_heatmap(fold_df, out_dir),
        fig_noise_curve(noise_df, out_dir),
    ]
    print(json.dumps({"generated": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
