# PIR-Net: Physics-Informed Resampling and Asymmetric Fusion Network

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/) 
[![.NET](https://img.shields.io/badge/.NET-10.0-purple)](https://dotnet.microsoft.com/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-green)]() 

> **Official Repository:** https://github.com/bolt-phm/PIR-Net

---

##  Introduction
PIR-Net is a state-of-the-art physics-informed deep learning framework designed specifically for **ultra-high frequency (1 MHz) bolt loosening detection** in industrial Prognostics and Health Management (PHM). 

In industrial environments, standard downsampling techniques (e.g., linear pooling) applied to 1 MHz signals often result in the aliasing or loss of micro-second duration impact transients. PIR-Net overcomes this by embedding **bolt impact dynamics** directly into the resampling layer and employing a heterogeneous feature fusion strategy.

**Key Achievements:**
* **150:1 Compression:** Integrates a **Physics-Informed Adaptive Resampling (PIAR)** module to compress 1 MHz signals while preserving 1–10 µs micro-transients.
* **Noise Robustness:** Utilizes **Asymmetric Cross-Modal Attention (ACMA)** to fuse time-domain and frequency-domain features, achieving superior performance in low SNR environments.
* **High Accuracy:** Validated on a six-class benchmark dataset, achieving **95.00% classification accuracy** with an inference speed exceeding **66 FPS**.

##  Industrial Deployment Software (BoltDetectionGUI)
To bridge the gap between academic research and industrial application, this repository includes **BoltDetectionGUI**, a professional Windows desktop application developed in **C# (.NET 10.0)**.

### Core Capabilities
#### 1. Real-Time USB Inference
The software implements a high-performance asynchronous I/O layer to communicate with USB Data Acquisition (DAQ) hardware. It enables **millisecond-level end-to-end latency**, visualizing the raw signal waveform and prediction confidence scores in real-time.

#### 2. Soft-Voting Ensemble Engine
Designed for harsh industrial environments, the GUI supports an **ensemble inference mode**. Users can load multiple trained model weights (`.pth`). The system applies a soft-voting mechanism to aggregate probability distributions, reducing false alarms caused by random noise.

#### 3. Automated Batch Testing
Includes a built-in generalization testing module. Users can import large-scale historical datasets, and the software will automatically generate Confusion Matrices, Precision/Recall Reports, and Failed Case Logs.

> **Note:** The C# source code is located in the `BoltDetectionGUI/` directory.

##  Physics-Informed Methodology
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

##  Performance Benchmarks
The framework was evaluated on the **Bolt Loosening Dataset (Exp. 222)**, covering states from **Severe Loose** to **Over Tight**.

| Metric | Performance | Notes |
| :--- | :--- | :--- |
| **Accuracy** | **95.00%** | State-of-the-art on test set |
| **Inference Speed** | **> 66 FPS** | ~15.0ms Latency (End-to-End) |
| **Safety Precision** | **99.80%** | Near-zero false alarms for dangerous states |
| **Noise Robustness** | **+56.7%** | Relative improvement at 0 dB SNR vs. ResNet |

##  Problem Analysis

### Core Challenges in Ultra-High Frequency Bolt Loosening Detection

#### 1. **Dimensionality Curse**
- **Problem:** 1 MHz sampling generates 1,000,000 data points per second, creating computational bottlenecks
- **Conventional Approach:** Standard downsampling (e.g., linear averaging) at 150:1 ratio
- **Limitation:** Critical micro-transients (1-10 µs) are averaged out, losing diagnostic information
- **Our Solution:** Physics-Informed Adaptive Resampling (PIAR) with hybrid pooling

#### 2. **Information Loss in Time-Frequency Representations**
- **Problem:** STFT-based spectrograms discard phase information and local gradients
- **Limitation:** Transition states (50-70% torque) exhibit subtle changes in time-frequency gradients
- **Our Solution:** 5-channel heterogeneous representation preserving gradients and raw morphology

#### 3. **Modality Imbalance in Multimodal Fusion**
- **Problem:** Spectral features dominate temporal features in conventional fusion
- **Limitation:** Noisy industrial environments degrade signal quality, requiring compensation
- **Our Solution:** Asymmetric Cross-Modal Attention (ACMA) with denoising retrieval

### Key Insights from Experimental Results

#### 1. **Transition States Are Not the Primary Challenge**
Contrary to expectations, the **Transition** class achieves F1-score of **0.9799**, indicating effective discrimination.

#### 2. **Secure vs. Over-Tight: The True Difficulty**
**Secure (F1=0.897)** and **Over-Tight (F1=0.907)** classes represent the performance bottleneck. Once securely tightened, further torque increases produce minimal acoustic impedance changes, making differentiation challenging even at 1 MHz sampling.

#### 3. **Safety-Critical Reliability**
**Severe Loose** class achieves **99.80% precision**, ensuring near-zero false alarms for dangerous conditions—critical for industrial safety applications.

##  Detailed Code Deployment

### Step 1: Environment Setup

#### Python Environment (Anaconda)
```bash
# Create and activate conda environment
conda create -n pirnet python=3.9
conda activate pirnet

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional dependencies
pip install numpy scipy opencv-python scikit-learn matplotlib seaborn tqdm thop tensorboard
```

#### .NET Environment (for GUI)
```bash
# Install .NET 10.0 SDK
# Download from: https://dotnet.microsoft.com/download

# Verify installation
dotnet --version
```

### Step 2: Dataset Preparation

#### Option 1: Download from Releases
```bash
# Download data.zip from GitHub Releases
wget https://github.com/bolt-phm/PIR-Net/releases/download/v1.0/data.zip

# Extract to project root
unzip data.zip -d ./data

# Verify structure
ls ./data/*.npy | head -5
```

#### Option 2: Git LFS Clone
```bash
# Clone repository with Git LFS
git lfs install
git clone https://github.com/bolt-phm/PIR-Net.git
cd PIR-Net
git lfs pull
```

#### Update Configuration
Edit `config.json` to match your data path:
```json
"data": {
  "data_dir": "./data",
  "case_ids": ["SevereLoose", "Loose", "Critical", "Transition", "Secure", "OverTight"],
  "num_classes": 6,
  "batch_size": 32,
  "image_size": [224, 224]
}
```

### Step 3: Model Training

#### Basic Training
```bash
python train.py

# Training with custom configuration
python train.py --exp_dir ./experiments/222
```

#### Key Training Parameters
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `epochs` | 100 | Total training epochs |
| `learning_rate` | 1e-4 | Initial learning rate |
| `batch_size` | 32 | Batch size for training |
| `label_smoothing_factor` | 0.1 | Smoothing factor for label smoothing |
| `combined_loss_ratio` | 0.8 | Ratio of LabelSmoothing in combined loss |
| `mixup_alpha` | 0.2 | Alpha parameter for MixUp augmentation |

### Step 4: Inference and Evaluation

#### Standard Inference
```bash
# Evaluate on test set
python generalization.py

# Output includes confusion matrix and classification report
# Results saved to: ./logs/final_generalization_matrix.png
```

#### Noise Robustness Testing
```bash
# Test with specific SNR level
export FORCE_SNR=0  # 0dB SNR
python generalization_all.py

# Batch testing across multiple SNR levels
python run_batch_kgr.py

# Comprehensive robustness benchmark
python run_robustness_benchmark.py
```

#### Efficiency Profiling
```bash
# Measure FLOPs, parameters, and inference speed
python measure_efficiency.py

# Expected output:
# FLOPs (计算量): 8.514G
# Params (参数量): 47.292M
# FPS (Throughput): 66.67 samples/sec
```

### Step 5: Ensemble Model Deployment

#### Prepare Ensemble Models
1. Train multiple models with different configurations
2. Save best checkpoints as `.pth` files
3. Update ensemble configuration:

```json
"inference": {
  "ensemble_models": [
    "./checkpoints/model_222.pth",
    "./checkpoints/model_223.pth",
    "./checkpoints/model_224.pth"
  ]
}
```

#### Run Ensemble Inference
```bash
# Ensemble inference uses soft voting by default
python generalization.py

# For distributed robustness testing
python robustness_worker.py --config_path ./222/config.json
```

### Step 6: GUI Software Deployment

#### Build from Source
```bash
cd BoltDetectionGUI

# Restore NuGet packages
dotnet restore

# Build in Release mode
dotnet publish -c Release -r win-x64 --self-contained

# Output: ./bin/Release/net10.0/win-x64/publish/BoltDetectionGUI.exe
```

#### Direct Download
1. Download `BoltDetectionGUI_v1.0.zip` from [Releases](https://github.com/bolt-phm/PIR-Net/releases)
2. Extract to any directory
3. Run `BoltDetectionGUI.exe`
4. **Note:** Requires .NET Desktop Runtime 10.0+

### Step 7: Troubleshooting Common Issues

#### Issue 1: CUDA Out of Memory
**Solution:** Reduce batch size in `config.json`:
```json
"data": {
  "batch_size": 16  # Reduce from 32
}
```

#### Issue 2: Data Loading Errors
**Solution:** Check data path and file format:
```bash
# Verify .npy files exist
ls ./data/SevereLoose/*.npy | wc -l

# Check file structure
python -c "import numpy as np; data = np.load('./data/SevereLoose/0001.npy'); print(data.shape)"
```

#### Issue 3: Import Errors
**Solution:** Install missing dependencies:
```bash
pip install --upgrade -r requirements.txt

# Or manually install specific packages
pip install scipy==1.11.0 opencv-python==4.8.0
```

#### Issue 4: Model Convergence Problems
**Solution:** Adjust learning rate and loss function:
```json
"train": {
  "learning_rate": 5e-5,  # Reduce learning rate
  "combined_loss_ratio": 0.9,  # Increase LabelSmoothing ratio
  "class_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Adjust for class imbalance
}
```

##  Advanced Configuration Guide

### Architecture Modifications

#### Change Image Encoder (model.py)
```python
# Modify config.json to switch encoder types
"modality": {
  "image_model": {
    "type": 1,  # 0=SimpleCNN, 1=ResNet50, 2=ResNet101
    "out_dim": 256,
    "pretrained": true
  }
}
```

#### Change Signal Encoder
```python
"modality": {
  "signal_model": {
    "type": 2,  # 0=SimpleRNN, 1=SimpleLSTM, 2=Hybrid CNN-LSTM-Transformer
    "in_channels": 1,
    "embed_dim": 256,
    "nhead": 8,
    "num_layers": 2,
    "dropout": 0.5
  }
}
```

#### Change Fusion Strategy
```python
"modality": {
  "fusion": {
    "type": 2,  # 0=Concat+Linear, 1=Attention, 2=MultiHeadAttention
    "num_heads": 4,
    "dropout": 0.5
  }
}
```

### Data Augmentation Settings

```python
"augment": {
  "signal": {
    "params": {
      "amplitude_scale_p": 0.5,
      "amplitude_scale_range": [0.8, 1.2],
      "noise_p": 0.5,
      "noise_amp": 0.01,
      "random_cutout_p": 0.0
    }
  },
  "image": {
    "color_jitter_p": 0.0,
    "brightness": 0.1,
    "contrast": 0.1,
    "random_erasing_p": 0.0,
    "erasing_area_range": [0.02, 0.1],
    "erasing_ratio_range": [0.3, 3.3]
  }
}
```

##  Project Structure
```text
PIR-Net/
├── BoltDetectionGUI/              # C# Desktop Application (.NET 10.0)
│   ├── Form1.cs                   # Main GUI interface
│   ├── Program.cs                 # Entry point
│   └── PytorchRunner.cs           # Python inference wrapper
├── config.json                   # Main configuration file
├── dataset.py                    # Physics-Informed Adaptive Resampling
│   └── SmartResampledDataset     # Multi-scale dataset class
├── model.py                      # Multi-modal network architecture
│   ├── MultiModalNet             # Main model class
│   ├── get_image_encoder()       # Image encoder factory
│   ├── get_signal_encoder()      # Signal encoder factory
│   └── get_fusion_module()       # Fusion module factory
├── train.py                      # Training pipeline
│   ├── CombinedLSLoss            # Combined loss function
│   ├── train_epoch()             # Training loop
│   └── validate()                # Validation loop
├── generalization.py             # Standard inference engine
│   └── EnsembleModel             # Ensemble model wrapper
├── generalization_all.py         # Noise robustness evaluation
├── robustness_worker.py          # Distributed robustness testing
├── run_batch_kgr.py              # Batch experiment runner
├── run_robustness_benchmark.py   # Comprehensive benchmark
├── measure_efficiency.py         # Performance profiling
├── run_all.sh                    # Automated training script
└── README.md                     # This documentation file

data/                             # Dataset directory
├── SevereLoose/                  # Class 0: <10% torque
├── Loose/                        # Class 1: 10-30% torque
├── Critical/                     # Class 2: 30-50% torque
├── Transition/                   # Class 3: 50-70% torque
├── Secure/                       # Class 4: 70-90% torque
└── OverTight/                    # Class 5: >100% torque
```

##  Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{PIRNet2026,
  title   = {PIR-Net: Physics-Informed Resampling and Asymmetric Fusion Network for Ultra-High Frequency Bolt Loosening Detection},
  author  = {Bolt-PHM Team},
  journal = {Neurocomputing},
  year    = {2026}
}
```

---

##  Acknowledgements

This research was supported by:
- National Natural Science Foundation of China [Grant No. 62373372]
- National Innovative Entrepreneurship Undergraduate Training Programme [Grant No. 202510489026]
- National Innovative Entrepreneurship Undergraduate Training Programme [Grant No. 202410489004]

We extend our sincere appreciation to the anonymous reviewers and editors for their valuable contributions.
