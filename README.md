# PIR-Net

Physics-informed deep learning for bolt loosening detection from 1 MHz vibration signals.

This repository is organized for **two real use cases**:

1. **Research reproduction**: run the full PIR-Net ablation suite and reviewer-requested baselines.
2. **Practical deployment**: use a Windows GUI helper tool (`BoltDetectionGUI`) for configuration and assisted inference workflows.

---

## Why this repo exists (pain points we solve)

If you have ever tried to reproduce bolt-loosening papers, you probably met these problems:

- Scripts are spread across many folders and difficult to map to paper experiments.
- Data preprocessing differs silently between methods, making comparisons unfair.
- Reviewer-requested baselines are missing or not runnable.
- New users cannot tell what to run first.
- Industrial users need a simple GUI, not just Python scripts.

This repo addresses those issues by:

- grouping experiments by type,
- explicitly separating **PIR-Net experiments** and **non-PIR baselines**,
- providing a Python bridge for GUI workflows,
- adding a beginner-friendly usage guide,
- shipping a Windows installer for the helper GUI.

---

## Repository layout

```text
PIRNet_OpenSource_Root/
├─ README.md
├─ docs/
│  └─ USAGE_GUIDE.md
├─ experiments/
│  ├─ pirnet_ablation/
│  │  ├─ 022/ 122/ 202/ 212/ 220/ 221/ 222/
│  │  └─ scripts/
│  └─ external_baselines/
│     ├─ core/
│     ├─ 301/ 302/ 303/ 304/ 305/ 306/
│     ├─ scripts/
│     └─ run_baselines.py
├─ BoltDetectionGUI/
│  ├─ src/                          # C# source code (.NET WinForms)
│  ├─ release/
│  │  ├─ BoltDetection_setup.exe    # installable helper tool package
│  │  └─ SHA256SUMS.txt
│  └─ README.md
├─ inference_engine.py              # GUI bridge script (required by GUI)
├─ tools/
│  ├─ update_data_dir.py
│  ├─ run_paper_pipeline.py
│  └─ publish_gui_release.ps1
└─ paper_support/
   ├─ py/
   └─ py_round2_figures/
```

---

## Quick start for beginners (0-to-run)

## Step 1: Prepare environment

### Python environment

Recommended: Python 3.10+ (tested with 3.12), PyTorch with CUDA.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pandas scikit-learn matplotlib seaborn opencv-python tqdm tensorboard
```

### Dataset

This repository is **code-only**. Dataset files are not bundled.

Expected data format:
- class folders (`case1` ... `case16`)
- each folder contains `.npy` run files

---

## Step 2: Point all experiments to your dataset path

From repository root:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data
```

This updates `data.data_dir` in all experiment `config.json` files.

---

## Step 3A: Run PIR-Net main experiment (example: Exp 222)

```bash
cd experiments/pirnet_ablation/222
python train.py --exp_dir .
python generalization.py --exp_dir .
```

---

## Step 3B: Run external baselines (301-306)

```bash
cd experiments/external_baselines
python run_baselines.py --experiments 301 302 303 304 305 306
```

Optional with post-evaluation:

```bash
python run_baselines.py --experiments 301 302 303 304 305 306 --with_generalization
```

---

## Step 4: (Optional) Distributed multi-server execution

PIR-Net split scripts:
- `experiments/pirnet_ablation/scripts/serverA.sh`
- `experiments/pirnet_ablation/scripts/serverB.sh`
- `experiments/pirnet_ablation/scripts/serverC.sh`
- `experiments/pirnet_ablation/scripts/serverD.sh`

Baseline split scripts:
- `experiments/external_baselines/scripts/serverA.sh`
- `experiments/external_baselines/scripts/serverB.sh`
- `experiments/external_baselines/scripts/serverC.sh`
- `experiments/external_baselines/scripts/serverD.sh`

See [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) for full commands.

---

## BoltDetectionGUI helper tool

`BoltDetectionGUI` is a Windows helper application for:

- project config editing,
- selecting Python runtime,
- assisted batch inference runs,
- optional USB mode workflow.

## Install GUI

Use installer:

- [BoltDetectionGUI/release/BoltDetection_setup.exe](BoltDetectionGUI/release/BoltDetection_setup.exe)

Verify checksum:

- [BoltDetectionGUI/release/SHA256SUMS.txt](BoltDetectionGUI/release/SHA256SUMS.txt)

## Important runtime requirement

The GUI expects a Python script named `inference_engine.py` in the selected project path.
This repository already includes it at root:

- [inference_engine.py](inference_engine.py)

So for easiest usage, in GUI set **Project Path** to this repository root (or another folder containing `config.json` + `inference_engine.py`).

---

## Publish GUI installer as GitHub release asset

You have two options:

### Option A: Manual on GitHub web

1. Create tag/release on GitHub.
2. Upload:
   - `BoltDetectionGUI/release/BoltDetection_setup.exe`
   - `BoltDetectionGUI/release/SHA256SUMS.txt`

### Option B: GitHub CLI script

```powershell
pwsh tools/publish_gui_release.ps1 -Tag v1.0.0 -Title "BoltDetectionGUI v1.0.0" -Notes "Windows installer for BoltDetectionGUI"
```

Prerequisite: `gh` installed and authenticated (`gh auth login`).

---

## Reproducibility notes

- PIR-Net and external baselines are separated by design.
- External baselines are configured to avoid PIR-specific preprocessing, matching reviewer constraints.
- If you change preprocessing policy, document it in your experiment report.

---

## Common issues and fixes

### 1) `Repository not found` when pushing
- Ensure remote URL and GitHub account are correct.
- Re-authenticate credentials if needed.

### 2) `dubious ownership` in Git

```bash
git config --global --add safe.directory E:/Desktop/GPT_Codex/PIRNet_OpenSource_Root
```

### 3) JSON BOM error (`Unexpected UTF-8 BOM`)
- Keep configs in UTF-8/UTF-8-SIG.
- This repo already uses BOM-tolerant loading where needed.

### 4) Linux script shebang errors
- Convert scripts to LF line endings if needed.

### 5) Low GPU utilization
- Increase `batch_size` and `num_workers` in each experiment config according to your VRAM/CPU.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

If this repository helps your research or deployment, please cite your PIR-Net paper and this repository URL.

