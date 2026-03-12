# Zenodo Cross-Condition Benchmark (`401`-`404`)

This directory provides a standalone benchmark for binary structural-state classification on Zenodo record `15516419` (`10.5281/zenodo.15516419`).

## 1. Label Definition

1. Class `0`: Baseline condition.
2. Class `1`: Damaged / loosened condition.

## 2. Included Experiments

1. `401`: WDCNN
2. `402`: InceptionTime-like
3. `403`: ResNet1D
4. `404`: CNN-BiLSTM-Attention

All models use 1D multi-channel signal input and support GPU training.

## 3. Dataset Placement

Place extracted benchmark files under:

```text
experiments/zenodo_generalization/data/zenodo_15516419/
```

The loader scans `.mat` files recursively in the defined dynamic-test folders.

## 4. Pre-Run Validation

```bash
python inspect_dataset.py --exp_dir ./401
```

## 5. Execution

Standard run:

```bash
python run_experiments.py --experiments 401 402 403 404 --with_generalization
```

Cross-validation run:

```bash
python run_cv_experiments.py --experiments 401 402 403 404 --with_generalization
```

## 6. Multi-Server Scripts

Scripts are available in `scripts/`, including:

1. `serverA.sh`-`serverD.sh`
2. `serverA_cv.sh`-`serverD_cv.sh`
3. `collect_results.sh`, `clean_all.sh`, `preflight.sh`

Typical launch:

```bash
ROOT=$(pwd) bash scripts/serverA.sh
```

## 7. Output Artifacts

Per experiment, outputs are written under `logs/` (for example metrics JSON, reports, confusion matrices, and run summaries). Runtime artifacts are excluded from source control by default.
