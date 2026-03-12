# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cm_baseline",
        type=str,
        default=str(
            _repo_root()
            / "重做2"
            / "serverA"
            / "022"
            / "logs"
            / "022_split-temporal_a-0p7_c-150_seed-3407"
            / "confusion_matrix_clean.png"
        ),
    )
    ap.add_argument(
        "--cm_ours",
        type=str,
        default=str(
            _repo_root()
            / "重做2"
            / "serverD"
            / "222"
            / "logs"
            / "222_split-temporal_a-0p7_c-150_seed-3407"
            / "confusion_matrix_clean.png"
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(_repo_root() / "lw2" / "figures" / "fig_rev2_confusion_clean_testset_exp022_vs_exp222.png"),
    )
    args = ap.parse_args()

    p_base = Path(args.cm_baseline)
    p_ours = Path(args.cm_ours)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    im0 = mpimg.imread(p_base)
    im1 = mpimg.imread(p_ours)

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].imshow(im0)
    axes[0].set_title("Baseline (Exp 022)  |  Test Split (Clean)")
    axes[0].axis("off")
    axes[1].imshow(im1)
    axes[1].set_title("PIR-Net (Exp 222)  |  Test Split (Clean)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    print(f"[OK] wrote: {out}")


if __name__ == "__main__":
    main()

