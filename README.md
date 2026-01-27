# PIR-Net: Physics-Informed Resampling Network for Bolt Loosening Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Task](https://img.shields.io/badge/Task-Fault_Diagnosis-green)]()

> **Project Repository:** [https://github.com/bolt-phm/PIR-Net](https://github.com/bolt-phm/PIR-Net)

## 📖 Introduction
The **PIR-Net** framework is a state-of-the-art multi-modal deep learning solution engineered for industrial bolt loosening detection. It addresses the critical challenge of **variable operating speeds** and **low Signal-to-Noise Ratio (SNR)** environments.

By integrating a **Physics-Informed Smart Resampling** mechanism with a **Dual-Branch Attention Fusion** architecture, PIR-Net extracts robust features from both 1D vibration signals and 2D Time-Frequency representations, achieving superior generalization capabilities across different working conditions.

## 🏗️ Core Architecture & Modules

### 1. Physics-Informed Data Engine (`dataset.py`)
The data pipeline is designed to preserve physical signal integrity during preprocessing.

- **Smart Adaptive Resampling:** Unlike traditional linear interpolation, this algorithm dynamically adjusts signal length while preserving critical transient impulses using a hybrid pooling strategy:
  $$ S_{out} = 0.7 \cdot \text{Max}(S_{in}) + 0.3 \cdot \text{Mean}(S_{in}) $$
  This ensures that high-frequency fault signatures are not lost during downsampling.

- **5-Channel STFT Generation:** Converts 1D signals into high-dimensional visual features:
  1. **Amplitude Spectrogram** (dB scale)
  2. **Frequency Gradient** (Edge detection)
  3. **Time Gradient** (Transient detection)
  4. **Energy Map** ($S^2$)
  5. **Raw Time-Domain View**
  All channels undergo **Robust Normalization** (3-sigma truncation) to suppress outliers.

- **On-the-Fly Noise Injection:** Simulates harsh industrial environments by injecting Additive White Gaussian Noise (AWGN) at random SNR levels (e.g., -5dB to 25dB) during training.

### 2. Multi-Modal Network Architecture (`model.py`)
The model employs a dual-stream encoder design with late fusion.

| Branch | Component | Description |
| :--- | :--- | :--- |
| **Image Encoder** | **ResNet50/101** | Modified input layer for 5-channel tensors. Extracts spatial Time-Frequency patterns. |
| **Signal Encoder** | **Hybrid SOTA** | Combines **Multi-Scale CNN** (local features), **Bi-LSTM** (temporal dependencies), and **Transformer Encoder** (global context). |
| **Fusion Layer** | **Attention** | Uses Multi-Head Attention to dynamically weigh the importance of visual vs. temporal modalities. |

### 3. Advanced Optimization Strategy (`train.py`)
To handle class imbalance and prevent overfitting, PIR-Net utilizes a **Combined Loss Landscape**:

$$ \mathcal{L}_{total} = \lambda \cdot \mathcal{L}_{LabelSmoothing} + (1-\lambda) \cdot \mathcal{L}_{Focal} $$

- **Label Smoothing:** Prevents the model from becoming over-confident on noisy labels.
- **Focal Loss:** Forces the model to focus on hard-to-classify samples.
- **Mixup Augmentation:** Linear interpolation of sample pairs ($x = \lambda x_i + (1-\lambda)x_j$) to smooth decision boundaries.

## 📂 Project Structure
```text
PIR-Net/
├── config.json              # Centralized configuration (Hyperparameters, Paths, Loss settings)
├── data.zip                 # Compressed source dataset (Requires extraction)
├── dataset.py               # Data pipeline: Smart Resampling, Physics Noise Injection, STFT
├── model.py                 # Neural Architecture: ResNet, Transformer, LSTM, Attention Fusion
├── train.py                 # Training Loop: Mixup, Combined Loss, Early Stopping
├── generalization.py        # Standard Inference: Basic accuracy testing
├── generalization_all.py    # Robustness Testing: SNR sweep (-5dB to 20dB) and ensemble evaluation
├── robustness_worker.py     # Batch Worker: Backend script for automated experiments
├── run_batch_kgr.py         # Experiment Orchestrator: Runs full testing suites across multiple models
├── measure_efficiency.py    # Profiler: Calculates FLOPs, Params, and Inference FPS
└── data_cleaning_step1.py   # Data Auditor: Detects mislabeled samples via high-loss analysis
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install torch torchvision numpy scipy opencv-python scikit-learn matplotlib seaborn tqdm thop tensorboard
```

### 2. Data Preparation
The dataset is provided as `data.zip`. Unzip it and map the path in `config.json`.
```bash
unzip data.zip -d ./data_set
```
Then update `config.json`:
```json
"data": {
    "data_dir": "./data_set",
    "case_ids": ["SevereLoose", "Loose", "Critical", "Transition", "Secure", "OverTight"]
}
```

### 3. Training & Evaluation
| Task | Command | Description |
| :--- | :--- | :--- |
| **Train** | `python train.py` | Starts training with params from config.json. Logs to TensorBoard. |
| **Test** | `python generalization.py` | Runs standard evaluation on the test split. |
| **Robustness** | `export FORCE_SNR=0 && python generalization_all.py` | Tests model performance under specific noise levels (e.g., 0dB). |
| **Audit** | `python data_cleaning_step1.py` | Scans dataset for potentially mislabeled or corrupted files. |

## 📝 Citation
If you use this code or methodology in your research, please cite:
```bibtex
@article{PIRNet2026,
  title={PIR-Net: A Multi-Modal Framework for Bolt Loosening Detection with Physics-Informed Resampling},
  author={Bolt-PHM Team},
  year={2026},
  journal={GitHub Repository}
}
```