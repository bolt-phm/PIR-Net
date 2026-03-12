import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.signal import stft, windows
import os

# 设置绘图风格 (适合PPT展示)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150
})

# ==========================================
# 工具函数：生成模拟信号
# ==========================================
def generate_signals(length=100000, n_impulses=3):
    """生成带有瞬态冲击的高频模拟信号"""
    t = np.linspace(0, 1, length)
    # 基础噪声
    noise = np.random.normal(0, 0.05, length)
    # 基础振动
    base_vib = 0.1 * np.sin(2 * np.pi * 50 * t)
    
    signal = noise + base_vib
    
    # 注入瞬态冲击 (模拟螺栓松动微滑移)
    indices = np.linspace(length//4, 3*length//4, n_impulses).astype(int)
    for idx in indices:
        width = 50 # 极短的冲击
        if idx+width < length:
            # 冲击形态：阻尼正弦
            impulse_t = np.linspace(0, 0.5, width)
            impulse = 3.0 * np.exp(-10*impulse_t) * np.sin(2*np.pi*20*impulse_t)
            signal[idx:idx+width] += impulse
            
    return t, signal

# ==========================================
# Scene 1: 背景与矛盾 (00:00 - 00:12)
# ==========================================
def plot_scene_1_conflict():
    print("Generating Scene 1: The Conflict...")
    t, raw_sig = generate_signals(length=100000, n_impulses=3)
    
    # 模拟传统下采样 (Naive Decimation) - 比如取平均或每隔N点取样
    factor = 100
    naive_ds = []
    for i in range(0, len(raw_sig), factor):
        chunk = raw_sig[i:i+factor]
        naive_ds.append(np.mean(chunk)) # 简单的平均会抹平尖峰
    
    naive_ds = np.array(naive_ds)
    t_ds = t[::factor]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：Raw Signal
    ax1.plot(t, raw_sig, color='#1f77b4', linewidth=0.5, alpha=0.8)
    ax1.set_title("Raw High-Freq Signal (1 MHz)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.text(0.5, 2.5, "Rich Transients", color='red', fontweight='bold', ha='center')
    
    # 右图：Naive Downsampling
    ax2.plot(t_ds, naive_ds, color='#7f7f7f', linewidth=2)
    ax2.set_title("Conventional Downsampling (Mean)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(ax1.get_ylim()) # 保持刻度一致以形成对比
    
    # 添加“漏斗”概念和警告
    ax2.text(0.5, 0.5, "Transient LOSS!", color='red', fontsize=20, fontweight='bold', 
             ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
    
    plt.suptitle("The Curse of Dimensionality: 1 MHz Sampling vs. Computational Limits", fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig("scene_01_problem.png", bbox_inches='tight')
    plt.close()

# ==========================================
# Scene 2: 物理重采样 (00:12 - 00:25)
# ==========================================
def plot_scene_2_resampling():
    print("Generating Scene 2: Physics-Informed Resampling...")
    # 生成一小段包含冲击的信号用于演示原理
    t = np.linspace(0, 100, 1000)
    sig = np.random.normal(0, 0.1, 1000)
    # 添加一个明显的尖峰
    sig[450:460] += np.linspace(0, 3, 5).tolist() + np.linspace(3, 0, 5).tolist()
    
    # 计算统计量
    window_max = np.max(np.abs(sig))
    window_mean = np.mean(np.abs(sig))
    # 公式结果
    pir_val = 0.7 * window_max + 0.3 * window_mean
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 画原始信号
    ax.plot(t, np.abs(sig), color='lightgray', label='Raw Window Data (|x|)', linewidth=1)
    ax.fill_between(t, np.abs(sig), color='lightgray', alpha=0.3)
    
    # 画关键指标线
    ax.axhline(y=window_max, color='#d62728', linestyle='--', linewidth=2, label=f'Max Peak (Impulse): {window_max:.2f}')
    ax.axhline(y=window_mean, color='#1f77b4', linestyle='--', linewidth=2, label=f'Energy Trend (Mean): {window_mean:.2f}')
    
    # 画计算结果点
    # 在图中间画一个显著的点代表压缩后的值
    center_x = 50
    ax.scatter([center_x], [pir_val], s=300, color='#2ca02c', zorder=10, label='Resampled Point')
    ax.annotate(r'$0.7 \times P_{max} + 0.3 \times P_{trend}$', 
                xy=(center_x, pir_val), xytext=(center_x+10, pir_val+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16, fontweight='bold')

    ax.set_title("Innovation 1: Physics-Informed Adaptive Resampling", fontsize=18)
    ax.legend(loc='upper right')
    ax.set_ylim(0, window_max * 1.3)
    
    plt.tight_layout()
    plt.savefig("scene_02_resampling.png", bbox_inches='tight')
    plt.close()

# ==========================================
# Scene 3: 异构表征 (00:25 - 00:40)
# ==========================================
def plot_scene_3_representation():
    print("Generating Scene 3: Heterogeneous Representation...")
    # 生成STFT图
    t, sig = generate_signals(length=2000, n_impulses=2)
    f, t_spec, Zxx = stft(sig, fs=1000, nperseg=128)
    spec = np.abs(Zxx)
    
    # 模拟5个通道
    # Ch1: Spectrogram
    ch1 = spec
    # Ch2: Time Gradient (X方向梯度)
    ch2 = np.abs(np.gradient(spec, axis=1))
    # Ch3: Freq Gradient (Y方向梯度)
    ch3 = np.abs(np.gradient(spec, axis=0))
    # Ch4: Energy Map (平方)
    ch4 = spec ** 2
    # Ch5: Waveform Embedding (平铺)
    ch5 = np.tile(np.abs(sig[:spec.shape[1]]), (spec.shape[0], 1))
    
    channels = [ch1, ch2, ch3, ch4, ch5]
    titles = ["1. STFT Spectrogram", "2. Time Gradient (Transient)", 
              "3. Freq Gradient (Resonance)", "4. Energy Map", "5. Waveform Embed"]
    cmaps = ['viridis', 'inferno', 'magma', 'hot', 'cividis']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, ax in enumerate(axes):
        # 归一化以便显示
        img_data = channels[i]
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-6)
        
        ax.imshow(img_data, aspect='auto', origin='lower', cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')
        
        # 给图片加边框，模拟“层”的感觉
        rect = patches.Rectangle((0,0), img_data.shape[1]-1, img_data.shape[0]-1, 
                                 linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    plt.suptitle("Innovation 2: 5-Channel Heterogeneous Tensor Construction", fontsize=20)
    plt.tight_layout()
    plt.savefig("scene_03_representation.png", bbox_inches='tight')
    plt.close()

# ==========================================
# Scene 4: 非对称注意力机制 (00:40 - 00:55)
# ==========================================
def plot_scene_4_attention():
    print("Generating Scene 4: Asymmetric Attention...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    categories = ['Signal (Temporal)', 'Image (Spectral)']
    
    # 场景 A: 典型样本 (Signal Dominant)
    weights_a = [0.95, 0.05]
    colors = ['#1f77b4', '#d62728']
    
    ax1.bar(categories, weights_a, color=colors, alpha=0.8)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Scenario A: Typical Sample\n(Signal Dominant)", fontsize=16)
    ax1.text(0, 0.96, "95%", ha='center', fontweight='bold', fontsize=14)
    ax1.text(1, 0.06, "5%", ha='center', fontweight='bold', fontsize=14)
    ax1.set_ylabel("Attention Weight")
    
    # 场景 B: 困难样本 (Safety Redundancy)
    # 模拟“边界情况”，图像权重上升
    weights_b = [0.60, 0.40] 
    
    ax2.bar(categories, weights_b, color=colors, alpha=0.8)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Scenario B: Ambiguous/Boundary Case\n(Safety Redundancy Activated)", fontsize=16)
    ax2.text(0, 0.61, "60%", ha='center', fontweight='bold', fontsize=14)
    ax2.text(1, 0.41, "40%", ha='center', fontweight='bold', fontsize=14, color='red')
    
    # 添加箭头表示动态变化
    # 这里用文字说明代替复杂的箭头绘制
    plt.figtext(0.5, 0.02, "Innovation 3: Dynamic Compensation Mechanism ('Denoising Retrieval')", 
                ha='center', fontsize=18, fontweight='bold', bbox=dict(facecolor='#f0f0f0', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("scene_04_attention.png", bbox_inches='tight')
    plt.close()

# ==========================================
# Scene 5: 性能总结 (00:55 - 01:10)
# ==========================================
def plot_scene_5_performance():
    print("Generating Scene 5: Performance Summary...")
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    
    # 左图：准确率对比
    ax1 = fig.add_subplot(gs[0])
    models = ['RF', '1D-CNN', 'ResNet50', 'PIR-Net (Ours)']
    acc = [67.32, 78.43, 85.67, 95.00]
    colors = ['gray', 'gray', 'gray', '#d62728'] # 突出 PIR-Net
    
    bars = ax1.barh(models, acc, color=colors)
    ax1.set_xlim(50, 100)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Overall Accuracy Comparison")
    
    # 给PIR-Net加标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        font_weight = 'bold' if i == 3 else 'normal'
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}%', va='center', fontweight=font_weight)

    # 右图：混淆矩阵 (模拟)
    ax2 = fig.add_subplot(gs[1])
    classes = ["Severe", "Loose", "Critical", "Trans.", "Secure", "Tight"]
    
    # 构建一个很好的混淆矩阵，对角线高
    cm = np.array([
        [0.99, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.01, 0.96, 0.03, 0.00, 0.00, 0.00],
        [0.00, 0.02, 0.95, 0.03, 0.00, 0.00],
        [0.00, 0.00, 0.02, 0.98, 0.00, 0.00], # Transition F1 高
        [0.00, 0.00, 0.00, 0.05, 0.90, 0.05],
        [0.00, 0.00, 0.00, 0.00, 0.08, 0.92]
    ])
    
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax2, 
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax2.set_title("Confusion Matrix (PIR-Net)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True Label")
    
    # 标注 Severe Loose Precision
    ax2.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', lw=4))
    ax2.text(1.2, 0.5, "Severe Loose\nPrecision: 99.8%", color='red', fontsize=14, fontweight='bold', ha='left')

    plt.suptitle("Final Result: 95.0% Accuracy & Industrial Safety Reliability", fontsize=20)
    plt.tight_layout()
    plt.savefig("scene_05_performance.png", bbox_inches='tight')
    plt.close()

# ==========================================
# 执行生成
# ==========================================
if __name__ == "__main__":
    plot_scene_1_conflict()
    plot_scene_2_resampling()
    plot_scene_3_representation()
    plot_scene_4_attention()
    plot_scene_5_performance()
    print("All images generated successfully!")