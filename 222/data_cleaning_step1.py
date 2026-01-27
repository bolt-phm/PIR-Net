import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 引用核心模块
from dataset import create_dataloaders
from model import build_model

# 配置环境，防止 Tensorboard 报错
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def get_per_sample_loss(output, target):
    """
    计算每个样本的独立 Loss (不求平均)。
    这里统一使用标准 CrossEntropy，因为它最能直观反映模型对正确类别的预测信心。
    Loss 越高 -> 模型认为这个样本越离谱。
    """
    return F.cross_entropy(output, target, reduction='none')

def main():
    # ---------------------------------------------------------
    # 1. 初始化与配置
    # ---------------------------------------------------------
    config_file = 'config.json'
    if not os.path.exists(config_file):
        print("❌ 错误: 当前目录下找不到 config.json")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    model_dir = config['train']['model_dir']
    log_dir = config['train']['log_dir']
    
    # 结果保存目录
    cleaning_dir = os.path.join(log_dir, "data_cleaning_results")
    os.makedirs(cleaning_dir, exist_ok=True)

    print(f"🚀 开始脏数据扫描...")
    print(f"🔧 使用设备: {device}")

    # ---------------------------------------------------------
    # 2. 加载数据 (开启 return_meta=True)
    # ---------------------------------------------------------
    print("📂 正在加载训练集 (开启元数据返回)...")
    
    # 调用 dataset.py 创建加载器
    # 注意：train_loaders 是一个字典 {'SmartResample_F150': DataLoader}
    train_loaders, _, _ = create_dataloaders(config, return_meta=True)
    
    loader_key = 'SmartResample_F150'
    
    # 【关键修复】空数据检查，防止 KeyError
    if loader_key not in train_loaders or not train_loaders[loader_key]:
        print("\n❌ 严重错误: DataLoader 为空！未能加载到任何数据。")
        print("💡 可能的原因：")
        print("   1. config.json 中的 'data_dir' 路径不对。")
        print("   2. config.json 中的 'fusion_map' 或 'case_ids' 与实际文件夹名不匹配。")
        print("   3. 文件夹存在，但里面没有 .npy 文件。")
        print("👉 请检查路径设置后重试。")
        return

    data_loader = train_loaders[loader_key]
    print(f"✅ 加载完成: {len(data_loader.dataset)} 个样本")

    # ---------------------------------------------------------
    # 3. 加载模型
    # ---------------------------------------------------------
    best_model_path = os.path.join(model_dir, config['train']['best_model_name'])
    
    # 如果找不到最佳模型，尝试找最后的断点
    if not os.path.exists(best_model_path):
        print(f"⚠️ 警告: 找不到最佳模型 {best_model_path}，尝试加载 checkpoint_last.pth")
        best_model_path = os.path.join(model_dir, "checkpoint_last.pth")
    
    if not os.path.exists(best_model_path):
        print("❌ 错误: 没有可用的模型权重文件，请先运行 train.py！")
        return

    print(f"🤖 加载模型权重: {best_model_path}")
    model = build_model(config).to(device)
    
    # 加载权重
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    model.eval()

    # ---------------------------------------------------------
    # 4. 执行扫描
    # ---------------------------------------------------------
    results = []
    
    print("🔍 正在逐个样本分析 Loss 和 置信度...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Scanning"):
            if batch is None: continue
            
            # dataset.py 的 return_meta=True 会返回 4 个值
            # 确保 dataset.py 是最新版
            try:
                img, sig, lbl, meta_info = batch
            except ValueError:
                print("❌ dataset.py 返回值数量不对，请确保使用的是最新的 dataset.py！")
                return
            
            img = img.to(device)
            sig = sig.to(device)
            lbl = lbl.to(device)
            
            # 前向传播
            output = model(img, sig)
            
            # 1. 计算每个样本的 Loss
            losses = get_per_sample_loss(output, lbl)
            
            # 2. 计算概率和预测
            probs = F.softmax(output, dim=1)
            confidences, predictions = torch.max(probs, 1)
            
            # 3. 计算熵 (Entropy) - 衡量不确定性
            log_probs = torch.log(probs + 1e-9)
            entropy = -torch.sum(probs * log_probs, dim=1)
            
            # 4. 收集数据
            for i in range(len(lbl)):
                # meta_info[i] 格式: "filename_offset"
                # 直接保存原始 meta_info，不做假设分割，保证完整性
                
                res_item = {
                    'meta_info': meta_info[i],
                    'true_label': lbl[i].item(),
                    'pred_label': predictions[i].item(),
                    'confidence': confidences[i].item(),
                    'loss': losses[i].item(),
                    'entropy': entropy[i].item(),
                    'is_correct': (lbl[i] == predictions[i]).item()
                }
                results.append(res_item)

    # 转换为 DataFrame 方便处理
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("❌ 警告：没有扫描到任何结果。")
        return

    # ---------------------------------------------------------
    # 5. 分析与筛选
    # ---------------------------------------------------------
    
    # === 策略 A: 标注错误嫌疑 (High Confidence Errors) ===
    # 预测错了，但是置信度还特别高 (> 0.8) -> 极有可能是数据标错了
    df_wrong = df[df['is_correct'] == False]
    suspicious_mislabeled = df_wrong[df_wrong['confidence'] > 0.8].sort_values(by='confidence', ascending=False)
    
    # === 策略 B: 极度困难/垃圾数据 (Top Loss) ===
    # 无论预测对错，Loss 最高的那 5% 数据 -> 可能是纯噪声或坏数据
    top_n = int(len(df) * 0.05) # 筛选前 5%
    if top_n < 1: top_n = 1
    suspicious_hard = df.sort_values(by='loss', ascending=False).head(top_n)
    
    # ---------------------------------------------------------
    # 6. 保存报告
    # ---------------------------------------------------------
    
    # 保存 CSV
    mislabeled_path = os.path.join(cleaning_dir, 'suspicious_mislabeled.csv')
    hard_path = os.path.join(cleaning_dir, 'suspicious_high_loss.csv')
    
    suspicious_mislabeled.to_csv(mislabeled_path, index=False)
    suspicious_hard.to_csv(hard_path, index=False)
    
    # 打印统计摘要
    print("\n" + "="*60)
    print("📊 脏数据扫描报告")
    print("="*60)
    print(f"总扫描样本数: {len(df)}")
    print(f"总体准确率 (Train): {df['is_correct'].mean():.4f}")
    print("-" * 60)
    print(f"🚩 疑似标注错误 (High Confidence Errors): {len(suspicious_mislabeled)} 个")
    print(f"   (预测错误且置信度 > 0.8，可能是文件放错文件夹了)")
    print(f"   -> 详情已保存至: {mislabeled_path}")
    if len(suspicious_mislabeled) > 0:
        print("   -> Top 5 嫌疑犯:")
        # 仅显示关键列
        cols = ['meta_info', 'true_label', 'pred_label', 'confidence']
        print(suspicious_mislabeled[cols].head(5).to_string(index=False))
    
    print("-" * 60)
    print(f"🚩 疑似垃圾/困难数据 (Top 5% Loss): {len(suspicious_hard)} 个")
    print(f"   (可能是纯噪声、全0数据或极端瞬态)")
    print(f"   -> 详情已保存至: {hard_path}")
    print("-" * 60)

    # ---------------------------------------------------------
    # 7. 可视化 Loss 分布 (辅助决策)
    # ---------------------------------------------------------
    try:
        plt.figure(figsize=(10, 6))
        # 绘制直方图
        sns.histplot(data=df, x='loss', hue='is_correct', bins=50, kde=True, palette={True: 'green', False: 'red'})
        plt.title('Loss Distribution: Correct vs Wrong Predictions')
        plt.xlabel('Loss Value')
        plt.ylabel('Count')
        # 画出 Top 5% 的阈值线
        threshold = suspicious_hard['loss'].min()
        plt.axvline(x=threshold, color='black', linestyle='--', label=f'Top 5% Threshold ({threshold:.2f})')
        plt.legend()
        plt.savefig(os.path.join(cleaning_dir, 'loss_distribution_analysis.png'))
        print(f"📈 Loss 分布图已保存: {os.path.join(cleaning_dir, 'loss_distribution_analysis.png')}")
    except Exception as e:
        print(f"⚠️ 绘图失败 (可能是缺少图形库): {e}")
    
    # ---------------------------------------------------------
    # 8. 操作建议
    # ---------------------------------------------------------
    print("\n💡 下一步建议:")
    print("1. 打开 'suspicious_mislabeled.csv'。")
    print("   重点检查 'meta_info' 对应的文件。这些文件极有可能被放错了文件夹。")
    print("2. 编写脚本根据 'meta_info' 画出波形图。")
    print("   - 如果是全黑/全噪图 -> 删！")
    print("   - 如果波形正常 -> 保留（这是困难样本）。")

if __name__ == '__main__':
    main()