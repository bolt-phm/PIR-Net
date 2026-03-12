# Usage Guide (Code-Only Release, Dataset Excluded)

This package is prepared for GitHub open-source release and intentionally **does not include any dataset files**.

## 1. Package Layout

```text
PIRNet_OpenSource_Root/
├─ README.md                         # Kept from the original repository (unchanged for now)
├─ LICENSE
├─ docs/
│  └─ USAGE_GUIDE.md                 # This document
├─ experiments/
│  ├─ pirnet_ablation/               # Main PIR-Net paper experiments
│  │  ├─ 022/ 122/ 202/ 212/ 220/ 221/ 222/
│  │  │  ├─ config.json
│  │  │  ├─ train.py
│  │  │  ├─ dataset.py
│  │  │  ├─ model.py
│  │  │  ├─ generalization.py
│  │  │  └─ run_robustness_benchmark.py
│  │  └─ scripts/                    # Multi-server split scripts for PIR-Net pipeline
│  │     ├─ serverA.sh
│  │     ├─ serverB.sh
│  │     ├─ serverC.sh
│  │     └─ serverD.sh
│  └─ external_baselines/            # Reviewer-requested baseline experiments (no PIR preprocessing)
│     ├─ core/                       # Shared baseline framework
│     ├─ 301/ 302/ 303/ 304/ 305/ 306/
│     │  ├─ config.json
│     │  ├─ train.py
│     │  ├─ dataset.py
│     │  ├─ model.py
│     │  └─ generalization.py
│     ├─ scripts/
│     │  ├─ serverA.sh
│     │  ├─ serverB.sh
│     │  ├─ serverC.sh
│     │  ├─ serverD.sh
│     │  └─ merge_results.py
│     ├─ run_baselines.py
│     └─ README.md
├─ tools/
│  ├─ update_data_dir.py             # Batch-update data_dir in all config.json files
│  ├─ run_paper_pipeline.py
│  ├─ run_batch_kgr.py
│  ├─ run_robustness_benchmark.py
│  ├─ measure_efficiency.py
│  ├─ generalization.py
│  ├─ generalization_all.py
│  ├─ robustness_worker.py
│  ├─ train.py
│  ├─ dataset.py
│  ├─ model.py
│  └─ config.json
└─ paper_support/
   ├─ py_round2_figures/             # Figure-generation scripts used in round-2 manuscript revision
   └─ py/
```

---

## 2. Environment Requirements

Recommended:
- Python 3.10+ (tested with 3.12)
- PyTorch + CUDA (GPU training strongly recommended)
- torchvision
- numpy, scipy, pandas, scikit-learn
- matplotlib, seaborn, opencv-python
- tqdm, tensorboard

Example installation (Linux):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pandas scikit-learn matplotlib seaborn opencv-python tqdm tensorboard
```

---

## 3. Dataset Preparation (Required)

Because datasets are excluded from this release, prepare your `.npy` dataset directory first.

Expected class folders (default naming used by configs):

```text
<YOUR_DATA_DIR>/
├─ case1/
├─ case2/
...
└─ case16/
```

Each class folder should contain run files like `caseX_runYY.npy`.

---

## 4. One-Time Path Update for All Experiments

From package root:

```bash
cd PIRNet_OpenSource_Root
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data
```

Optional (if you also use generalization directory fields):

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data --generalization_dir /absolute/path/to/generalization_data
```

This updates every `config.json` under:
- `experiments/pirnet_ablation/*`
- `experiments/external_baselines/*`

---

## 5. Run Main PIR-Net Ablation Experiments (022–222)

### 5.1 Single experiment run

```bash
cd experiments/pirnet_ablation/222
python train.py --exp_dir .
```

Run generalization/evaluation for the same experiment:

```bash
python generalization.py --exp_dir .
```

### 5.2 Run all main ablation experiments sequentially

```bash
cd experiments/pirnet_ablation
for EXP in 022 122 202 212 220 221 222; do
  python "$EXP/train.py" --exp_dir "$EXP"
done
```

### 5.3 Multi-server split (as used in your paper workflow)

From package root:

```bash
cd PIRNet_OpenSource_Root
chmod +x experiments/pirnet_ablation/scripts/serverA.sh experiments/pirnet_ablation/scripts/serverB.sh experiments/pirnet_ablation/scripts/serverC.sh experiments/pirnet_ablation/scripts/serverD.sh
```

Server A:

```bash
ROOT=$(pwd) bash experiments/pirnet_ablation/scripts/serverA.sh
```

Server B:

```bash
ROOT=$(pwd) bash experiments/pirnet_ablation/scripts/serverB.sh
```

Server C:

```bash
ROOT=$(pwd) bash experiments/pirnet_ablation/scripts/serverC.sh
```

Server D:

```bash
ROOT=$(pwd) bash experiments/pirnet_ablation/scripts/serverD.sh
```

Notes:
- These scripts call `tools/run_paper_pipeline.py` logic and are designed for distributed execution.
- The scripts preserve original experiment logic and parameter grouping.

---

## 6. Run External Baseline Experiments (301–306)

All baselines are intentionally configured as **non-PIR preprocessing** baselines to keep algorithm assumptions faithful.

### 6.1 Single baseline run

```bash
cd experiments/external_baselines/301
python train.py --exp_dir .
python generalization.py --exp_dir .
```

### 6.2 Run all baselines sequentially

```bash
cd experiments/external_baselines
python run_baselines.py --experiments 301 302 303 304 305 306
```

With post-training generalization:

```bash
python run_baselines.py --experiments 301 302 303 304 305 306 --with_generalization
```

### 6.3 Multi-server split for baselines

```bash
cd PIRNet_OpenSource_Root/experiments/external_baselines
chmod +x scripts/serverA.sh scripts/serverB.sh scripts/serverC.sh scripts/serverD.sh
```

Server A:

```bash
ROOT=$(pwd) bash scripts/serverA.sh
```

Server B:

```bash
ROOT=$(pwd) bash scripts/serverB.sh
```

Server C:

```bash
ROOT=$(pwd) bash scripts/serverC.sh
```

Server D:

```bash
ROOT=$(pwd) bash scripts/serverD.sh
```

Merge collected outputs:

```bash
python scripts/merge_results.py
```

---

## 7. Output Locations

Typical outputs:
- `checkpoints/...` (model weights)
- `logs/...` (metrics, confusion matrices, reports)
- `results/...` (server-partition result bundles)

All output folders are generated automatically during training/evaluation.

---

## 8. Reproduce Figure Scripts (Optional)

Figure-generation scripts used in manuscript revision are in:
- `paper_support/py_round2_figures/`
- `paper_support/py/`

Run each script according to its expected CSV/result inputs.

---

## 9. Practical Notes

- If your Linux shell reports `#!/usr/bin/env: No such file or directory`, normalize script line endings to LF.
- If JSON loading fails with BOM-related errors, ensure configs are UTF-8/UTF-8-SIG.
- If GPU usage is low, increase batch size and `num_workers` in `config.json` based on available VRAM/CPU.

---

## 10. BoltDetectionGUI Helper Tool

This repository also includes an optional Windows helper application:
- `BoltDetectionGUI/src/` (source)
- `BoltDetectionGUI/release/BoltDetection_setup.exe` (installer)

Runtime bridge requirement:
- The GUI expects `inference_engine.py` in the selected project path.
- This repository ships `inference_engine.py` at the root.

For easiest operation, set the GUI **Project Path** to the repository root.

## 11. README status

The repository `README.md` has been updated to match the latest code structure, GUI packaging, and beginner deployment flow.
