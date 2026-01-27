# PIR-Net: Physics-Informed Resampling & Asymmetric Fusion Network

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)
[![.NET](https://img.shields.io/badge/.NET-10.0-purple)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-green)]()

> **Project Repository:** [https://github.com/bolt-phm/PIR-Net](https://github.com/bolt-phm/PIR-Net)

## 📖 Introduction
PIR-Net is a robust deep learning framework designed for **Ultra-High Frequency (1 MHz) Bolt Loosening Detection**. It addresses the critical trade-off between computational efficiency and transient feature preservation in industrial PHM.

By integrating a **Physics-Informed Adaptive Resampling** layer with an **Asymmetric Cross-Modal Attention** mechanism, PIR-Net achieves **150:1 compression** while retaining micro-transients (1-10µs) essential for early fault detection. The system has been validated to achieve **95.00% accuracy** on a 6-class dataset under varying noise conditions.

## 🖥️ BoltDetectionGUI Software
This repository includes a professional **Windows Desktop Application** developed in **C# (.NET 10.0)** for real-time industrial deployment.

### Key Features
- **🔌 Real-Time USB Inference:** Connects directly to DAQ hardware for live signal monitoring with ms-level latency. Displays prediction confidence and status instantly.
- **🎲 Ensemble Configuration:** Supports 'Soft-Voting' ensemble inference. Users can load multiple `.pth` weight files to boost robustness against noise.
- **📊 Batch Testing Mode:** One-click batch generalization testing. Automatically generates confusion matrices and performance reports inside the GUI.
- **⚙️ Full Parameter Control:** Modify Hyperparameters (Epochs, LR, Batch Size) and Network Architecture (ResNet type, Fusion type) via a user-friendly interface.

*(Source code located in `BoltDetectionGUI/` folder)*

## 🧠 Physics-Informed Mechanisms
### 1. Physics-Informed Smart Resampling
Conventional downsampling causes aliasing artifacts. We propose a dynamic pooling strategy driven by bolt dynamics (Impact Transients & Energy Shifts):
$$ S_{out} = 0.7 \cdot \text{Max}(|S_{in}|) + 0.3 \cdot \text{Mean}(|S_{in}|) $$
This hybrid approach ensures high-frequency impact signatures are preserved ($P_{max}$) while retaining global energy trends ($P_{energy}$).

### 2. Heterogeneous Gradient-Spectral Representation
To capture 'Transition' states (50-70% torque), we construct a **5-Channel Tensor** that explicitly encodes gradient information:
* **Ch1: Log-Spectrogram** ($S_{db}$)
* **Ch2: Time Gradient** ($\nabla_t S_{db}$, capturing transient onsets)
* **Ch3: Frequency Gradient** ($\nabla_f S_{db}$, capturing resonance shifts)
* **Ch4: Energy Map** ($S^2$)
* **Ch5: Time-Domain Embedding** (Raw waveform morphology)
This representation solves the phase-information loss problem inherent in standard STFT.

### 3. Asymmetric Cross-Modal Attention
We employ a **Signal-Dominant, Image-Compensating** fusion strategy. The attention mechanism calculates weights dynamically:
$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$
where $Q$ comes from visual features, while $K, V$ are concatenated from both modalities. This allows the stable spectral features to selectively retrieve complementary details from noisy temporal signals, serving as a safety redundancy.

## 🚀 Performance Benchmarks
Validated on the 6-class Bolt Loosening Dataset (SevereLoose to OverTight).

| Metric | Performance | Description |
| :--- | :--- | :--- |
| **Accuracy** | **95.00%** | State-of-the-art on Test Set (Exp 222) |
| **Inference Speed** | **> 66 FPS** | ~15.0ms Total Latency (End-to-End) |
| **Safety Precision** | **99.80%** | Near-zero false alarms for 'Severe Loose' states |
| **Robustness** | **+56.7%** | Relative improvement at 0dB SNR compared to baselines |

## 📦 Installation & Data Preparation

### 1. Requirements
```bash
pip install torch torchvision numpy scipy opencv-python scikit-learn matplotlib seaborn tqdm thop tensorboard
```

### 2. Dataset Setup
The dataset is compressed in `data.zip`. You must unzip it before training.
```bash
# Unzip to project root
unzip data.zip -d ./data_set
```
Then update `config.json`:
```json
"data": {
    "data_dir": "./data_set",
    "use_offline": true
}
```

## 🛠️ Usage (Python Pipeline)

### Train the Model
```bash
python train.py
```
* **Loss:** Combined Label Smoothing + Focal Loss
* **Logs:** Saved to `tf-logs/` (Viewable via TensorBoard)

### Robustness Testing
Test model performance under specific Noise (SNR) conditions:
```bash
# Test with 0dB Noise Injection
export FORCE_SNR=0 && python generalization_all.py
```

### Efficiency Profiling
Calculate FLOPs, Parameters, and Inference FPS:
```bash
python measure_efficiency.py
```

## 📂 Project Structure
```text
PIR-Net/
├── BoltDetectionGUI/        # C# Source Code for Desktop App
├── config.json              # Central Configuration
├── data.zip                 # Dataset (Compressed)
├── dataset.py               # Physics-Informed Resampling & STFT
├── model.py                 # Multi-Modal Network Architecture
├── train.py                 # Training Loop with Combined Loss
├── generalization.py        # Standard Inference Script
├── generalization_all.py    # Robustness Testing with SNR Injection
├── run_batch_kgr.py         # Batch Experiment Runner
└── measure_efficiency.py    # FLOPs/Params Profiler
```

## 📝 Citation
If you use this code or methodology in your research, please cite:
```bibtex
@article{PIRNet2026,
  title={PIR-Net: Physics-Informed Resampling and Asymmetric Fusion Network for Ultra-High Frequency Bolt Loosening Detection},
  author={Bolt-PHM Team},
  journal={Neurocomputing},
  year={2026}
}
```
