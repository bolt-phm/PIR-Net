# PGRF-Net Open Source

PGRF-Net is a physics-guided deep learning framework for bolt-loosening diagnosis from ultra-high-frequency (1 MHz) vibration signals.

## Overview

This repository provides a complete and reproducible project package, including:

1. PGRF-Net ablation experiments (`022`-`222`).
2. External baseline experiments (`301`-`307`) under a unified non-PIR protocol.
3. Public-dataset cross-condition generalization experiments (`401`-`404`, Zenodo baseline controls).
4. Feature-engineering controls on Zenodo (`501`-`503`) for round-2 supplementary analysis.
5. A Windows auxiliary validation GUI (`BoltDetectionGUI`).
6. Curated experiment-result bundles and figure-generation scripts.

## Repository Layout

```text
PIRNet_OpenSource_Root/
|-- README.md
|-- requirements.txt
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- CODE_OF_CONDUCT.md
|-- SECURITY.md
|-- data.zip
|-- docs/
|   |-- USAGE_GUIDE.md
|   `-- assets/
|-- experiments/
|   |-- pirnet_ablation/
|   |-- external_baselines/
|   `-- zenodo_generalization/
|-- experiment_results/
|-- BoltDetectionGUI/
|-- paper_support/
|-- tools/
`-- *.py (project-level runtime entry points)
```

## Data Provenance

`data.zip` is the in-house dataset used for PGRF-Net development.

1. Source: self-collected impact-vibration measurements from an FPGA-based acquisition setup.
2. Sampling rate: 1 MHz.
3. Storage format: `.npy` samples organized by `case1`-`case16`.
4. Availability: versioned in this repository via Git LFS.

The external benchmark used for cross-condition validation is Zenodo record `15516419` (`10.5281/zenodo.15516419`) and is not bundled in this repository.

Representative acquisition setup:

![FPGA-based data acquisition setup](docs/assets/fpga_acquisition_setup.png)

## Environment Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git lfs install
git lfs pull
```

## Data Path Configuration

Use the utility below to update experiment `config.json` files before training:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data
```

Optional generalization data path:

```bash
python tools/update_data_dir.py --root . --data_dir /absolute/path/to/your/data --generalization_dir /absolute/path/to/generalization_data
```

## Quick Start

PGRF-Net (`Exp 222`):

```bash
cd experiments/pirnet_ablation/222
python train.py --exp_dir .
python generalization.py --exp_dir .
```

External baselines (`Exp 301`-`307`):

```bash
cd experiments/external_baselines
python run_baselines.py --experiments 301 302 303 304 305 306 307
python run_baselines.py --experiments 301 302 303 304 305 306 307 --with_generalization
```

Zenodo cross-condition benchmark (`Exp 401`-`503`):

```bash
cd experiments/zenodo_generalization
python run_cv_experiments.py --experiments 401 402 403 404 501 502 503 \
  --with_generalization --generalization_experiments 404 501 \
  --snrs clean 10 5 0 --snr_repeats 3
```

Round-2 merged result package (from six-server union, complete coverage):

```text
experiment_results/zenodo_round2_cv_63x18/
```

The package contains:

1. `coverage_summary.json` (63/63 train runs, 18/18 generalization runs).
2. `train_manifest.csv` and `train_summary_by_experiment.csv`.
3. `generalization_file_manifest.csv`, `generalization_long.csv`, and `generalization_summary_by_group_snr.csv`.
4. `raw/` copied per-run artifacts used to build the summary tables.

## Validation and Tooling

Repository smoke check:

```bash
python -m tools.smoke_test --mode all --exp_dir experiments/pirnet_ablation/222
```

Auxiliary GUI installer:

1. `BoltDetectionGUI/release/BoltDetection_setup.exe`
2. `BoltDetectionGUI/release/SHA256SUMS.txt`

## Citation

Citation metadata is maintained in `CITATION.cff`. Use the GitHub "Cite this repository" panel to export BibTeX or APA entries.

## Governance

1. Contribution process: `CONTRIBUTING.md`
2. Community policy: `CODE_OF_CONDUCT.md`
3. Vulnerability handling: `SECURITY.md`

## License

This project is released under the MIT License (`LICENSE`).
