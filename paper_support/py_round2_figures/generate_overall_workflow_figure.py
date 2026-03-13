from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _repo_root() -> Path:
    # lw2/py_round2_figures/generate_overall_workflow_figure.py -> repo root
    return Path(__file__).resolve().parents[2]


def _draw_module(ax, x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="semibold",
        color="#1f2937",
        linespacing=1.25,
    )


def _draw_arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color="#4b5563",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def draw_overall_workflow(out_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax = plt.subplots(figsize=(19, 4.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    modules = [
        ("Raw 1 MHz\nVibration Signal", "#e0f2fe", "#0ea5e9"),
        ("Physics-Informed\nAdaptive Resampling", "#e0f7f1", "#10b981"),
        ("Heterogeneous Representation\n(Pseudo-image + 1D Signal)", "#fff7e6", "#f59e0b"),
        ("Dual-Branch\nFeature Encoding", "#eef2ff", "#6366f1"),
        ("Cross-Modal\nAttention Fusion", "#f5f3ff", "#8b5cf6"),
        ("Linear Classifier +\nLabel Smoothing Loss", "#fce7f3", "#ec4899"),
        ("Bolt-Loosening\nState Prediction", "#ecfdf5", "#22c55e"),
    ]

    n = len(modules)
    left_margin = 0.03
    right_margin = 0.03
    gap = 0.017
    total_width = 1.0 - left_margin - right_margin - gap * (n - 1)
    w = total_width / n
    h = 0.42
    y = 0.29

    centers = []
    x = left_margin
    for label, fc, ec in modules:
        _draw_module(ax, x, y, w, h, label, fc, ec)
        centers.append((x + w / 2.0, y + h / 2.0, x, x + w))
        x += w + gap

    mid_y = y + h / 2.0
    for i in range(n - 1):
        _, _, _, right_edge = centers[i]
        _, _, left_edge_next, _ = centers[i + 1]
        _draw_arrow(ax, right_edge + 0.004, mid_y, left_edge_next - 0.004, mid_y)

    ax.text(
        0.5,
        0.92,
        "Overall Workflow of PIR-Net for Bolt-Loosening Detection",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.11,
        "From raw high-frequency vibration input to final bolt-loosening state prediction",
        ha="center",
        va="center",
        fontsize=11,
        color="#374151",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig_overall_workflow_7modules.png"
    pdf_path = out_dir / "fig_overall_workflow_7modules.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    out_dir = _repo_root() / "lw2" / "figures"
    png_path, pdf_path = draw_overall_workflow(out_dir)
    print(f"[OK] wrote: {png_path}")
    print(f"[OK] wrote: {pdf_path}")


if __name__ == "__main__":
    main()
