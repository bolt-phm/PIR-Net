import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import platform

# --- 字体配置（防止中文乱码） ---
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] 
elif system_name == "Darwin": 
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# --- 风格配置 ---
# 使用更高级的配色方案
plt.style.use('seaborn-v0_8-whitegrid')
colors_blue = sns.color_palette("Blues", n_colors=6)
colors_red = sns.color_palette("Reds", n_colors=6)

OUTPUT_DIR = "ppt_images"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_plot(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"生成的图表: {filepath}")
    return filepath

# 1. 核心亮点：重采样对比图 (物理感知 vs 线性)
def draw_resampling_highlight():
    t = np.linspace(0, 100, 500)
    # 模拟一个带有稀疏冲击的信号
    signal = np.sin(t * 0.1) * 0.2 + np.random.normal(0, 0.02, 500)
    signal[200:205] += 2.5 # 添加冲击
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # 传统方法失败的展示
    ax1.plot(t, signal, color='lightgray', label='原始信号 (1MHz)')
    ax1.plot(t[::20], signal[::20], 'o--', color='gray', label='传统线性降采样')
    ax1.annotate('冲击丢失！\n(Aliasing)', xy=(t[202], 0.2), xytext=(t[202]+10, 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')
    ax1.set_title("痛点：传统方法导致高频特征丢失", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # PIR-Net 方法成功的展示
    smart_y = []
    smart_x = t[::20]
    for i in range(0, len(signal), 20):
        chunk = signal[i:i+20]
        val = 0.7 * np.max(np.abs(chunk)) + 0.3 * np.mean(np.abs(chunk))
        smart_y.append(val)
        
    ax2.plot(t, signal, color='lightgray', alpha=0.5)
    ax2.plot(smart_x, smart_y, 's-', color='#d62728', linewidth=2, label='PIR 物理感知重采样')
    ax2.annotate('特征完美保留', xy=(t[202], 2.0), xytext=(t[202]+10, 2.5),
                 arrowprops=dict(facecolor='#d62728', shrink=0.05), fontsize=12, color='#d62728', fontweight='bold')
    ax2.set_title("创新点：PIR 策略保留微弱冲击 (150:1 压缩率)", fontsize=14, fontweight='bold', color='#d62728')
    ax2.legend(loc='upper right')
    
    return save_plot(fig, "chart_resampling.png")

# 2. 成果展示：SOTA 对比柱状图
def draw_sota_bar():
    methods = ['RF', 'XGBoost', 'CNN-LSTM', 'ResNet-101', 'PIR-Net (Ours)']
    acc = [67.32, 71.85, 83.27, 88.92, 96.04]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, acc, color=['gray', 'gray', '#aec7e8', '#1f77b4', '#d62728'])
    
    ax.set_ylim(60, 100)
    ax.set_ylabel("准确率 Accuracy (%)", fontsize=12)
    ax.set_title("性能突破：远超现有 SOTA 方法", fontsize=16, fontweight='bold')
    
    # 标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    # 标注提升幅度
    ax.annotate(f'+7.12%', 
                xy=(4, 96.04), xytext=(3, 93),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2", color='black'),
                fontsize=12, fontweight='bold', color='#d62728')

    return save_plot(fig, "chart_sota.png")

# 3. 机理分析：动态补偿 (高光时刻)
def draw_mechanism_highlight():
    labels = ['常规样本\n(信号主导)', '困难样本\n(图像补偿)']
    signal_attn = [0.99, 0.52]
    image_attn = [0.01, 0.48]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.5
    
    ax.bar(labels, signal_attn, width, label='振动信号权重', color='#1f77b4', alpha=0.8)
    ax.bar(labels, image_attn, width, bottom=signal_attn, label='图像频谱权重', color='#d62728', alpha=0.9)
    
    ax.set_ylim(0, 1.2)
    ax.set_title("核心机理：动态补偿机制", fontsize=16, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    
    # 添加文字解释
    ax.text(0, 0.5, "平时依靠高频信号\n(效率优先)", ha='center', color='white', fontweight='bold', fontsize=11)
    ax.text(1, 0.25, "信号模糊时\n视觉介入", ha='center', color='white', fontweight='bold', fontsize=11)
    ax.text(1, 0.8, "安全冗余\n(Safety Net)", ha='center', color='white', fontweight='bold', fontsize=11)
    
    return save_plot(fig, "chart_mechanism.png")

if __name__ == "__main__":
    print("正在生成高清数据图表...")
    draw_resampling_highlight()
    draw_sota_bar()
    draw_mechanism_highlight()
    print("图表生成完毕！请继续运行 main_ppt_builder.py")