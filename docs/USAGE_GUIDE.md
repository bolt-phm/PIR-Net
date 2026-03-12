# Usage Guide (Dataset Included via Git LFS)

This guide describes environment setup, dataset handling, and experiment execution for PIR-Net.

## 1. Package Layout

```text
PIRNet_OpenSource_Root/
├─ README.md
├─ LICENSE
├─ docs/
│  └─ USAGE_GUIDE.md
├─ experiments/
│  ├─ pirnet_ablation/
│  └─ external_baselines/
├─ tools/
│  ├─ update_data_dir.py
│  ├─ run_paper_pipeline.py
│  ├─ run_batch_kgr.py
│  ├─ run_robustness_benchmark.py
│  ├─ measure_efficiency.py
│  └─ ...
└─ paper_support/
```

## 2. Environment Requirements

Recommended:

- Python 3.10+ (tested with 3.12)
- PyTorch + CUDA for training
- torchvision, numpy, scipy, pandas, scikit-learn
- matplotlib, seaborn, opencv-python, tqdm, tensorboard

Install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Dataset Preparation

Download LFS objects and extract `data.zip`:

```bash
git lfs install
git lfs pull
```

Expected class folders:

```text
<YOUR_DATA_DIR>/
├─ case1/
├─ case2/
...
└─ case16/
```

Each class folder should contain run files like `caseX_runYY.npy`.

## 4. One-Time Path Update for All Experiments

From package root:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data
```

Optional:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data --generalization_dir /absolute/path/to/generalization_data
```

## 5. Run PIR-Net Experiments (022-222)

Single experiment:

```bash
cd experiments/pirnet_ablation/222
python train.py --exp_dir .
python generalization.py --exp_dir .
```

## 6. Run External Baseline Experiments (301-306)

Single baseline:

```bash
cd experiments/external_baselines/301
python train.py --exp_dir .
python generalization.py --exp_dir .
```

All baselines:

```bash
cd experiments/external_baselines
python run_baselines.py --experiments 301 302 303 304 305 306
python run_baselines.py --experiments 301 302 303 304 305 306 --with_generalization
```

## 7. Multi-Server Execution (Optional)

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

## 8. Output Locations

Typical outputs:

- `checkpoints/...`
- `logs/...`
- `results/...`

## 9. Auxiliary GUI

The repository includes `BoltDetectionGUI` for assisted validation.

Runtime bridge requirement:

- `config.json`
- `inference_engine.py`
- trained weights in `checkpoints/...`

## 10. Practical Notes

- If shell scripts fail with `/usr/bin/env` errors, normalize line endings to LF.
- If JSON loading fails due BOM, save configs as UTF-8/UTF-8-SIG.
- If GPU usage is low, increase `batch_size` and `num_workers` in `config.json`.
