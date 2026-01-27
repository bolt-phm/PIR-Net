# PIR-Net: Physics-Informed Resampling and Asymmetric Fusion Network

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/) 
[![.NET](https://img.shields.io/badge/.NET-10.0-purple)](https://dotnet.microsoft.com/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-green)]() 

> **Official Repository:** https://github.com/bolt-phm/PIR-Net

---

## Introduction
PIR-Net is a state-of-the-art physics-informed deep learning framework designed specifically for **ultra-high frequency (1 MHz) bolt loosening detection** in industrial Prognostics and Health Management (PHM). 

In industrial environments, standard downsampling techniques (e.g., linear pooling) applied to 1 MHz signals often result in the aliasing or loss of micro-second duration impact transients. PIR-Net overcomes this by embedding **bolt impact dynamics** directly into the resampling layer and employing a heterogeneous feature fusion strategy.

**Key Achievements:**
* **150:1 Compression:** Integrates a **Physics-Informed Adaptive Resampling (PIAR)** module to compress 1 MHz signals while preserving 1–10 µs micro-transients.
* **Noise Robustness:** Utilizes **Asymmetric Cross-Modal Attention (ACMA)** to fuse time-domain and frequency-domain features, achieving superior performance in low SNR environments.
* **High Accuracy:** Validated on a six-class benchmark dataset, achieving **95.00% classification accuracy** with an inference speed exceeding **66 FPS**.

## Industrial Deployment Software (BoltDetectionGUI)
To bridge the gap between academic research and industrial application, this repository includes **BoltDetectionGUI**, a professional Windows desktop application developed in **C# (.NET 10.0)**.

### Core Capabilities
#### 1. Real-Time USB Inference
The software implements a high-performance asynchronous I/O layer to communicate with USB Data Acquisition (DAQ) hardware. It enables **millisecond-level end-to-end latency**, visualizing the raw signal waveform and prediction confidence scores in real-time.

#### 2. Soft-Voting Ensemble Engine
Designed for harsh industrial environments, the GUI supports an **ensemble inference mode**. Users can load multiple trained model weights (`.pth`). The system applies a soft-voting mechanism to aggregate probability distributions, reducing false alarms caused by random noise.

#### 3. Automated Batch Testing
Includes a built-in generalization testing module. Users can import large-scale historical datasets, and the software will automatically generate Confusion Matrices, Precision/Recall Reports, and Failed Case Logs.

> **Note:** The C# source code is located in the `BoltDetectionGUI/` directory.

## Physics-Informed Methodology
### 4.1 Physics-Informed Adaptive Resampling (PIAR)
Uniform downsampling introduces aliasing artifacts that obscure short-duration impact responses. PIR-Net adopts a physics-driven adaptive pooling strategy. The resampling weights are governed by the **maximum impact amplitude** ($P_{\max}$) and the **local energy density** ($P_{\text{energy}}$).

The downsampled signal $x_{ds}[i]$ is calculated as:

$$
x_{ds}[i] = \alpha \cdot \mathcal{P}_{\max}(w_i) + (1-\alpha) \cdot \mathcal{P}_{\text{energy}}(w_i)
$$

Where $\alpha=0.7$ is the balancing factor. The energy term is calculated as the **Root Mean Square (RMS)** to represent the vibrational energy within the window $w_i$:

$$
\mathcal{P}_{\text{energy}}(w_i) = \sqrt{\frac{1}{N} \sum_{j \in w_i} x_j^2}
$$

This hybrid formulation ensures that high-frequency impact signatures are preserved ($P_{\max}$) while retaining global energy trends ($P_{\text{energy}}$).

### 4.2 Heterogeneous Gradient–Spectral Representation
To characterize intermediate loosening states (e.g., 50–70% residual torque), PIR-Net constructs a **five-channel heterogeneous tensor**:
* **Ch 1: Log-Spectrogram ($S_{db}$):** Captures the fundamental time-frequency distribution.
* **Ch 2: Temporal Gradient ($\nabla_t S_{db}$):** Highlights the onset of transient impacts.
* **Ch 3: Frequency Gradient ($\nabla_f S_{db}$):** Detects resonance frequency shifts caused by stiffness degradation.
* **Ch 4: Energy Map ($S^2$):** Emphasizes high-energy regions.
* **Ch 5: Waveform Morphology:** A time-domain embedding that preserves phase information lost in STFT.

### 4.3 Asymmetric Cross-Modal Attention (ACMA)
PIR-Net employs a novel asymmetric fusion mechanism. We use the spectral features as the **Query ($Q$)** to retrieve complementary details from the time-domain **Key ($K$)** and **Value ($V$)** pairs:

$$
\text{Attention}(Q_f, K_{tf}, V_{tf}) = \text{softmax}\left( \frac{Q_f K_{tf}^{\top}}{\sqrt{d_k}} \right) V_{tf}
$$

## Performance Benchmarks
The framework was evaluated on the **Bolt Loosening Dataset (Exp. 222)**, covering states from **Severe Loose** to **Over Tight**.

| Metric | Performance | Notes |
| :--- | :--- | :--- |
| **Accuracy** | **95.00%** | State-of-the-art on test set |
| **Inference Speed** | **> 66 FPS** | ~15.0ms Latency (End-to-End) |
| **Safety Precision** | **99.80%** | Near-zero false alarms for dangerous states |
| **Noise Robustness** | **+56.7%** | Relative improvement at 0 dB SNR vs. ResNet |

## 📦 Detailed Deployment Guide

### Step 1: Environment Setup (Anaconda)
We recommend using Anaconda to manage dependencies. Open your terminal/command prompt:
```bash
# 1. Create a new environment (Python 3.8+ recommended)
conda create -n pirnet python=3.9

# 2. Activate the environment
conda activate pirnet

# 3. Install PyTorch (Adjust CUDA version based on your GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 2: Install Dependencies
Install the remaining required Python libraries:
```bash
pip install numpy scipy opencv-python scikit-learn matplotlib seaborn tqdm thop tensorboard
```

### Step 3: Dataset Preparation
The dataset is hosted via Git LFS. You can download it directly from the **Releases** page or clone it.
1. **Download** `data.zip` from the [Latest Release](https://github.com/bolt-phm/PIR-Net/releases).
2. **Unzip** the file into the project root directory.
```bash
unzip data.zip -d ./
# Ensure the folder structure is: ./data/SevereLoose, ./data/Loose, etc.
```
3. **Verify Config**: Open `config.json` and check:
```json
"data": {
  "data_dir": "./data",
  "use_offline": true
}
```

### Step 4: Running the Model
**To Train the Model:**
```bash
python train.py
# Logs will be saved to tf-logs/ and checkpoints to checkpoints/
```

**To Evaluate Performance (Batch Test):**
```bash
python generalization.py
```

### Step 5: Running the GUI Software
1. Go to the [Releases Page](https://github.com/bolt-phm/PIR-Net/releases).
2. Download `BoltDetectionGUI_v1.0.zip`.
3. Unzip and run `BoltDetectionGUI.exe` directly (No installation required).
4. **Note:** Ensure you have `.NET Desktop Runtime 10.0` (or latest) installed if it doesn't start.

## Project Structure
```text
PIR-Net/
├── BoltDetectionGUI/        # C# Source Code (.NET 10.0)
├── config.json              # Hyperparameter Configuration
├── data.zip                 # 1 MHz Dataset (Git LFS)
├── dataset.py               # PIAR Resampling & Preprocessing
├── model.py                 # PIR-Net Architecture (ACMA Fusion)
├── train.py                 # Training Loop
├── generalization.py        # Inference Engine
└── measure_efficiency.py    # Profiling Tools
```

## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{PIRNet2026,
  title   = {PIR-Net: Physics-Informed Resampling and Asymmetric Fusion Network for Ultra-High Frequency Bolt Loosening Detection},
  author  = {Bolt-PHM Team},
  journal = {Neurocomputing},
  year    = {2026}
}
```
