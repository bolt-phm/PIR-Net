# Zenodo Cross-Condition Benchmark (`401`-`503`)

This directory provides the standalone public-dataset benchmark used in revision experiments on Zenodo record `15516419` (`10.5281/zenodo.15516419`).

## 1. Label Definition

1. Class `0`: Baseline condition.
2. Class `1`: Damaged / loosened condition.

## 2. Included Experiments

1. `401`: WDCNN
2. `402`: InceptionTime-like
3. `403`: ResNet1D
4. `404`: CNN-BiLSTM-Attention
5. `501`: PGRF-Net pipeline (physics-guided downsampling)
6. `502`: Average-downsample control (A)
7. `503`: Decimate-based feature pipeline (C)

All models use 1D multi-channel signal input and support GPU training.

## 3. Dataset Placement

Place extracted benchmark files under:

```text
experiments/zenodo_generalization/data/zenodo_15516419/
```

The loader scans `.mat` files recursively in the dynamic-test folders.

## 4. Pre-Run Validation

```bash
python inspect_dataset.py --exp_dir ./401
python scripts/preflight_check.py
```

## 5. CV Execution (Full Round-2 Scope)

```bash
python run_cv_experiments.py --experiments 401 402 403 404 501 502 503 \
  --folds fold_d15_cfg22 fold_d16_cfg21 fold_d17_cfg21 \
  --seeds 2026 3407 4096 \
  --with_generalization --generalization_experiments 404 501 \
  --snrs clean 10 5 0 --snr_repeats 3
```

## 6. Six-Server Orchestration

Use the finalized split-balanced scripts in `scripts/`:

1. `server01_main_part1.sh`
2. `server02_main_part2.sh`
3. `server03_main_multiseed.sh`
4. `server04_sensitivity_and_leakage.sh`
5. `server05_robustness_pir.sh`
6. `server06_baselines_and_robustness.sh`
7. `progress_all.sh`

Typical launch:

```bash
ROOT=$(pwd) bash scripts/server01_main_part1.sh
```

Progress check:

```bash
ROOT=$(pwd) bash scripts/progress_all.sh
```

## 7. Output Artifacts

Per experiment, outputs are written under `logs/` and `checkpoints/` (metrics JSON, reports, confusion matrices, run summaries, and optional weights).
Runtime outputs are intentionally excluded from source control by default.
