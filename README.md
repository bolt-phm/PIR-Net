# PIR-Net

PIR-Net is a physics-informed deep learning framework for bolt-loosening diagnosis from ultra-high-frequency (1 MHz) vibration signals.

## Overview

This repository provides a complete and reproducible project package, including:

1. PIR-Net ablation experiments (`022`-`222`).
2. External baseline experiments (`301`-`307`) under a unified non-PIR protocol.
3. Public-dataset cross-condition generalization experiments (`401`-`404`, Zenodo).
4. A Windows auxiliary validation GUI (`BoltDetectionGUI`).
5. A curated experiment-result bundle and figure-generation scripts.

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

`data.zip` is the in-house dataset used for PIR-Net development.

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

PIR-Net (`Exp 222`):

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

Zenodo cross-condition benchmark (`Exp 401`-`404`):

```bash
cd experiments/zenodo_generalization
python run_experiments.py --experiments 401 402 403 404 --with_generalization
```

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
