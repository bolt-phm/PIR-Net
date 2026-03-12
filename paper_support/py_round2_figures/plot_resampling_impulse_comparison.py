# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def smart_resample_abs(raw_signal: np.ndarray, target_len: int, alpha: float) -> np.ndarray:
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    input_len = raw_signal.shape[0]
    if input_len < target_len:
        # Fallback: simple interpolation
        x = np.linspace(0, 1, input_len, endpoint=False)
        xi = np.linspace(0, 1, target_len, endpoint=False)
        return np.interp(xi, x, raw_signal).astype(np.float32)

    factor = max(1, input_len // target_len)
    valid_len = target_len * factor
    cropped = raw_signal[:valid_len]
    reshaped = cropped.reshape(target_len, factor)
    abs_sig = np.abs(reshaped)
    peak_hold = np.max(abs_sig, axis=1)
    energy_trend = np.mean(abs_sig, axis=1)
    return (alpha * peak_hold + (1.0 - alpha) * energy_trend).astype(np.float32)


def downsample_mean_abs(raw_signal: np.ndarray, target_len: int) -> np.ndarray:
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    factor = max(1, raw_signal.shape[0] // target_len)
    valid_len = target_len * factor
    x = np.abs(raw_signal[:valid_len]).reshape(target_len, factor)
    return x.mean(axis=1).astype(np.float32)


def downsample_max_abs(raw_signal: np.ndarray, target_len: int) -> np.ndarray:
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    factor = max(1, raw_signal.shape[0] // target_len)
    valid_len = target_len * factor
    x = np.abs(raw_signal[:valid_len]).reshape(target_len, factor)
    return x.max(axis=1).astype(np.float32)


def downsample_decimate_abs(raw_signal: np.ndarray, target_len: int) -> np.ndarray:
    raw_signal = np.asarray(raw_signal, dtype=np.float32)
    factor = max(1, raw_signal.shape[0] // target_len)
    idx = np.arange(0, target_len) * factor + factor // 2
    idx = np.clip(idx, 0, raw_signal.shape[0] - 1)
    return np.abs(raw_signal[idx]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npy",
        type=str,
        default=str(_repo_root() / "data" / "data" / "case1" / "case1_run01.npy"),
        help="Path to a raw 1MHz vibration file (npy).",
    )
    ap.add_argument("--target_len", type=int, default=8192)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--bins_radius", type=int, default=60)
    ap.add_argument(
        "--out",
        type=str,
        default=str(_repo_root() / "lw2" / "figures" / "fig_rev2_resampling_impulse_comparison_alpha0p7_ratio150.png"),
    )
    args = ap.parse_args()

    p = Path(args.npy)
    raw = np.load(p, mmap_mode="r")
    raw = np.asarray(raw, dtype=np.float32)
    target_len = int(args.target_len)
    alpha = float(args.alpha)

    # Find a salient impulse by max absolute amplitude.
    abs_raw = np.abs(raw)
    peak_idx = int(abs_raw.argmax())

    # Map to downsampled bin.
    factor = max(1, raw.shape[0] // target_len)
    bin_idx = peak_idx // factor
    lo = max(0, bin_idx - int(args.bins_radius))
    hi = min(target_len, bin_idx + int(args.bins_radius) + 1)

    # Raw window aligned to bins.
    raw_lo = lo * factor
    raw_hi = min(raw.shape[0], hi * factor)
    raw_win = abs_raw[raw_lo:raw_hi]

    # Downsample methods (all compare on abs magnitude)
    y_dec = downsample_decimate_abs(raw, target_len)[lo:hi]
    y_mean = downsample_mean_abs(raw, target_len)[lo:hi]
    y_max = downsample_max_abs(raw, target_len)[lo:hi]
    y_hyb = smart_resample_abs(raw, target_len, alpha=alpha)[lo:hi]

    # Peak retention (relative to raw peak in the aligned window)
    raw_peak = float(raw_win.max()) if raw_win.size else 1.0
    def ret(x): return float(np.max(x) / (raw_peak + 1e-9))

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), gridspec_kw={"height_ratios": [1.2, 1.0]})

    # Top: raw abs window
    axes[0].plot(raw_win, linewidth=0.8, color="#444444")
    axes[0].set_title(f"Raw signal (abs) window around peak  |  file={p.name}")
    axes[0].set_ylabel("|x(t)|")
    axes[0].grid(alpha=0.2)

    # Bottom: resampled abs sequences around the corresponding bins
    xs = np.arange(lo, hi)
    axes[1].plot(xs, y_dec, marker="o", markersize=3, linewidth=1.6, label=f"Decimate (ret={ret(y_dec):.2f})")
    axes[1].plot(xs, y_mean, marker="o", markersize=3, linewidth=1.6, label=f"MeanPool (ret={ret(y_mean):.2f})")
    axes[1].plot(xs, y_max, marker="o", markersize=3, linewidth=1.6, label=f"MaxPool (ret={ret(y_max):.2f})")
    axes[1].plot(xs, y_hyb, marker="o", markersize=3, linewidth=2.0, label=f"Hybrid α={alpha:.1f} (ret={ret(y_hyb):.2f})")
    axes[1].set_title("Compressed-domain impulse retention (abs magnitude)")
    axes[1].set_xlabel("Compressed sample index")
    axes[1].set_ylabel("value")
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=2, loc="upper right")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] wrote: {out}")


if __name__ == "__main__":
    main()

