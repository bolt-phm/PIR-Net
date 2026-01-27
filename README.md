# PIR-Net: Physics-Informed Resampling and Asymmetric Fusion Network

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)
[![.NET](https://img.shields.io/badge/.NET-10.0-purple)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-green)]()

> **Official Repository:** https://github.com/bolt-phm/PIR-Net

---

## 📖 Introduction
PIR-Net is a physics-informed deep learning framework for **ultra-high frequency (1 MHz) bolt loosening detection** in industrial prognostics and health management (PHM) systems.

Conventional vibration-based pipelines suffer from severe information loss when aggressive downsampling is applied to ultra-high frequency signals. PIR-Net addresses this limitation by explicitly embedding **bolt impact dynamics** into both the resampling process and the feature fusion strategy.

By integrating a **Physics-Informed Adaptive Resampling (PIAR)** module with an **Asymmetric Cross-Modal Attention (ACMA)** mechanism, PIR-Net achieves up to **150:1 temporal compression** while preserving micro-transient signatures (1–10 µs) that are critical for early-stage loosening identification. The proposed framework attains **95.00% classification accuracy** on a six-class bolt loosening dataset under varying noise conditions.

## 🖥️ Industrial Deployment Software (BoltDetectionGUI)
This repository includes a standalone **Windows desktop application** developed in **C# (.NET 10.0)** for real-time industrial deployment and operator-level interaction.

### Key Capabilities
- **🔌 Real-Time USB Inference:** Direct connection to DAQ hardware, enabling millisecond-level end-to-end inference latency with live prediction confidence visualization.
- **🎲 Ensemble-Based Inference:** Supports soft-voting ensemble strategies by loading multiple trained `.pth` model weights to improve robustness under severe noise conditions.
- **📊 Batch Generalization Evaluation:** Integrated batch testing mode with automatic generation of confusion matrices and quantitative performance reports.
- **⚙️ Configurable Training and Architecture Parameters:** Full control over training hyperparameters and network configurations via a graphical user interface.

> Source code is located in the `BoltDetectionGUI/` directory.

## 🧠 Physics-Informed Methodology
### 4.1 Physics-Informed Adaptive Resampling
Uniform downsampling introduces aliasing artifacts that obscure short-duration impact responses in ultra-high frequency vibration signals. To mitigate this effect, PIR-Net adopts a physics-driven adaptive pooling strategy guided by bolt impact transients and energy redistribution:

$$
S_{\text{out}} = 0.7 \cdot \max\left(|S_{\text{in}}|\right) + 0.3 \cdot mean\left(|S_{\text{in}}|\right)
$$

This hybrid formulation emphasizes high-amplitude transient responses while retaining global energy information. As a result, high-frequency impact signatures ($P_{\max}$) are preserved alongside long-term energy trends ($P_{\text{energy}}$), effectively suppressing aliasing-induced degradation.

### 4.2 Heterogeneous Gradient–Spectral Representation
To characterize intermediate loosening states (e.g., 50–70% residual torque), PIR-Net constructs a **five-channel heterogeneous representation** that jointly encodes spectral content and its local variations:

- **Channel 1:** Log-scaled spectrogram ($S_{\text{db}}$)
- **Channel 2:** Temporal gradient ($\nabla_t S_{\text{db}}$), highlighting transient onsets
- **Channel 3:** Frequency gradient ($\nabla_f S_{\text{db}}$), capturing resonance migration
- **Channel 4:** Energy map ($S^2$)
- **Channel 5:** Time-domain embedding (raw waveform morphology)

This representation alleviates the phase information loss inherent in standard STFT-based pipelines and improves sensitivity to subtle structural state transitions.

### 4.3 Asymmetric Cross-Modal Attention Fusion
PIR-Net employs an **asymmetric cross-modal attention** mechanism that prioritizes stable spectral representations while selectively compensating them with temporal details:

$$
Attention(Q, K, V) = softmax\left( \frac{QK^{\top}}{\sqrt{d_k}} \right)V
$$

The query $Q$ is derived from spectral-domain features, whereas the key–value pairs $(K, V)$ are constructed from concatenated spectral and temporal embeddings. This design enhances robustness under high-noise conditions and provides redundancy against unreliable time-domain measurements.

## 🚀 Performance Evaluation
The proposed framework is evaluated on a six-class bolt loosening dataset ranging from **Severe Loose** to **Over Tight** conditions.

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Accuracy** | **95.00%** | Test set performance (Exp. 222) |
| **Inference Speed** | **> 66 FPS** | ~15 ms end-to-end latency |
| **Safety Precision** | **99.80%** | Near-zero false alarms for critical loose states |
| **Noise Robustness** | **+56.7%** | Relative improvement at 0 dB SNR |

## 📦 Installation and Data Preparation

### Environment Requirements
```bash
pip install torch torchvision numpy scipy opencv-python scikit-learn matplotlib seaborn tqdm thop tensorboard
```

### Dataset Preparation
```bash
unzip data.zip -d ./data_set
```

Update `config.json` accordingly:
```json
"data": {
  "data_dir": "./data_set",
  "use_offline": true
}
```

## 🛠️ Usage

### Model Training
```bash
python train.py
```
- **Loss Function:** Label Smoothing + Focal Loss
- **Logging:** TensorBoard logs saved in `tf-logs/`

### Noise Robustness Evaluation
```bash
export FORCE_SNR=0
python generalization_all.py
```

### Computational Efficiency Profiling
```bash
python measure_efficiency.py
```

## 📂 Project Structure
```text
PIR-Net/
├── BoltDetectionGUI/        # Industrial desktop application (C#)
├── config.json             # Global configuration
├── data.zip                # Dataset archive
├── dataset.py              # Physics-informed resampling and STFT
├── model.py                # Multi-modal network architecture
├── train.py                # Training pipeline
├── generalization.py       # Standard inference
├── generalization_all.py   # Noise robustness evaluation
├── run_batch_kgr.py        # Batch experiment runner
└── measure_efficiency.py   # FLOPs and latency profiling
```

## 📝 Citation
If you use this repository or methodology in academic work, please cite:
```bibtex
@article{PIRNet2026,
  title   = {PIR-Net: Physics-Informed Resampling and Asymmetric Fusion Network for Ultra-High Frequency Bolt Loosening Detection},
  author  = {Bolt-PHM Team},
  journal = {Neurocomputing},
  year    = {2026}
}
```
