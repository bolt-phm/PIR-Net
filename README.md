# PIR-Net

Physics-informed deep learning framework for bolt loosening detection from 1 MHz vibration signals.

## 1. Overview

This repository provides the codebase used for PIR-Net experiments and reviewer-requested baseline comparisons in bolt loosening diagnosis. It is organized for reproducible research, controlled baseline evaluation, and assisted validation via a lightweight Windows GUI.

The repository intentionally excludes raw datasets from version control.

## 2. Scope

This project covers three technical workflows:

1. PIR-Net ablation and main-model training/evaluation.
2. External baseline benchmarking (configured without PIR-specific preprocessing to preserve baseline fidelity).
3. Assisted validation using `BoltDetectionGUI` as an orchestration front-end for Python inference.

## 3. Repository Structure

```text
PIRNet_OpenSource_Root/
├─ README.md
├─ LICENSE
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
│  ├─ src/
│  ├─ release/
│  │  ├─ BoltDetection_setup.exe
│  │  └─ SHA256SUMS.txt
│  └─ README.md
├─ tools/
│  ├─ update_data_dir.py
│  ├─ run_paper_pipeline.py
│  └─ publish_gui_release.ps1
├─ inference_engine.py
├─ train.py
├─ generalization.py
└─ dataset.py
```

## 4. System Requirements

Recommended runtime:

- Python 3.10+ (validated with 3.12)
- CUDA-capable GPU for training
- PyTorch, torchvision, torchaudio
- numpy, scipy, pandas, scikit-learn
- matplotlib, seaborn, opencv-python
- tqdm, tensorboard

Example installation:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pandas scikit-learn matplotlib seaborn opencv-python tqdm tensorboard
```

## 5. Dataset Preparation

Datasets are not distributed in this repository. Prepare a directory containing class subfolders (default: `case1` to `case16`) and `.npy` samples.

Reference layout:

```text
<YOUR_DATA_DIR>/
├─ case1/
├─ case2/
...
└─ case16/
```

## 6. Configure Data Paths

Before training, update all experiment configs:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data
```

Optional (if generalization path is required):

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data --generalization_dir /absolute/path/to/generalization_data
```

## 7. Running Experiments

### 7.1 PIR-Net Experiments

Single experiment example (`222`):

```bash
cd experiments/pirnet_ablation/222
python train.py --exp_dir .
python generalization.py --exp_dir .
```

### 7.2 External Baselines (301-306)

Sequential run:

```bash
cd experiments/external_baselines
python run_baselines.py --experiments 301 302 303 304 305 306
```

With generalization:

```bash
python run_baselines.py --experiments 301 302 303 304 305 306 --with_generalization
```

### 7.3 Distributed Multi-Server Execution

PIR-Net split scripts:

- `experiments/pirnet_ablation/scripts/serverA.sh`
- `experiments/pirnet_ablation/scripts/serverB.sh`
- `experiments/pirnet_ablation/scripts/serverC.sh`
- `experiments/pirnet_ablation/scripts/serverD.sh`

Baselines split scripts:

- `experiments/external_baselines/scripts/serverA.sh`
- `experiments/external_baselines/scripts/serverB.sh`
- `experiments/external_baselines/scripts/serverC.sh`
- `experiments/external_baselines/scripts/serverD.sh`

Detailed command examples are provided in `docs/USAGE_GUIDE.md`.

## 8. Output Artifacts

Typical generated outputs include:

- model checkpoints
- training/validation logs
- confusion matrices and generalization reports
- merged baseline result summaries

Output paths are controlled by each experiment's `config.json`.

## 9. BoltDetectionGUI (Validation Assistant)

`BoltDetectionGUI` is an auxiliary Windows validation assistant. It is designed to simplify project configuration and invoke Python-side inference/validation workflows.

Important scope clarification:

- It is an assisted verification interface.
- It is not a standalone DAQ/runtime engine.

Installer and integrity files:

- `BoltDetectionGUI/release/BoltDetection_setup.exe`
- `BoltDetectionGUI/release/SHA256SUMS.txt`

Runtime requirement:

- The selected project path must contain `config.json`, `inference_engine.py`, and trained checkpoints.

## 10. Publishing GUI Installer to GitHub Release

Manual approach:

1. Create a tag and release on GitHub.
2. Upload installer assets from `BoltDetectionGUI/release/`.

Scripted approach (GitHub CLI required):

```powershell
pwsh tools/publish_gui_release.ps1 -Tag v1.0.0 -Title "BoltDetectionGUI v1.0.0" -Notes "Windows installer for assisted validation"
```

## 11. Reproducibility Notes

- PIR-Net and external baselines are intentionally separated.
- Baseline experiments are configured to avoid PIR-specific preprocessing.
- Any deviation from default preprocessing or split strategy should be documented in reports.

## 12. Troubleshooting

### 12.1 Git safe-directory warning

```bash
git config --global --add safe.directory E:/Desktop/GPT_Codex/PIRNet_OpenSource_Root
```

### 12.2 JSON BOM parsing errors

Ensure `config.json` files are saved in UTF-8 or UTF-8-SIG.

### 12.3 Linux shebang issues

If shell scripts fail with `/usr/bin/env` errors, convert line endings to LF.

### 12.4 Low GPU utilization

Increase `batch_size` and `num_workers` in experiment configs according to hardware limits.

## 13. License

This repository is released under the MIT License. See `LICENSE` for details.

## 14. Citation

If you use this codebase in academic work, please cite the PIR-Net paper and include this repository URL.