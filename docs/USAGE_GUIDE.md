# PIR-Net Usage Guide

This document provides standardized setup and execution instructions for all experiments in this repository.

## 1. Prerequisites

1. Python 3.10+ (tested with Python 3.12).
2. CUDA-enabled PyTorch environment for training.
3. Git LFS for retrieving `data.zip`.

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git lfs install
git lfs pull
```

## 2. Dataset Preparation

### 2.1 In-house dataset (`data.zip`)

1. Extract `data.zip` to a local path.
2. Ensure the directory contains `case1`-`case16`.
3. Each case folder should include `.npy` run files.

### 2.2 External benchmark (Zenodo 15516419)

For `experiments/zenodo_generalization`, place extracted benchmark files under:

```text
experiments/zenodo_generalization/data/zenodo_15516419/
```

## 3. Global Path Update

Update all experiment configs in one pass:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/inhouse_data
```

Optional:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/inhouse_data --generalization_dir /absolute/path/to/extra_generalization_data
```

## 4. PIR-Net Ablation (`022`-`222`)

Example (`Exp 222`):

```bash
cd experiments/pirnet_ablation/222
python train.py --exp_dir .
python generalization.py --exp_dir .
```

## 5. External Baselines (`301`-`307`)

Run all baselines:

```bash
cd experiments/external_baselines
python run_baselines.py --experiments 301 302 303 304 305 306 307
python run_baselines.py --experiments 301 302 303 304 305 306 307 --with_generalization
```

Run a single baseline:

```bash
cd experiments/external_baselines/307
python train.py --exp_dir .
python generalization.py --exp_dir .
```

## 6. Zenodo Cross-Condition Generalization (`401`-`404`)

Pre-check:

```bash
cd experiments/zenodo_generalization
python inspect_dataset.py --exp_dir ./401
```

Run:

```bash
python run_experiments.py --experiments 401 402 403 404 --with_generalization
```

Optional cross-validation run:

```bash
python run_cv_experiments.py --experiments 401 402 403 404 --with_generalization
```

## 7. Multi-Server Launch Scripts

Available script sets:

1. `experiments/pirnet_ablation/scripts/`
2. `experiments/external_baselines/scripts/`
3. `experiments/zenodo_generalization/scripts/`

Use each script from the corresponding experiment root with:

```bash
ROOT=$(pwd) bash scripts/<server>.sh
```

## 8. Output Conventions

Training artifacts are generated in experiment-local folders (for example `logs/`, `checkpoints/`, and `results/`) and are excluded from source control by default.

## 9. Validation Utilities

Repository smoke validation:

```bash
python -m tools.smoke_test --mode all --exp_dir experiments/pirnet_ablation/222
```
