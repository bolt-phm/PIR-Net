import os
import json
import torch
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 引入自定义模块
from dataset import create_dataloaders
from model import build_model  # 确保 model.py 中的 build_model 可用

# 强制设置日志编码
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------------------------
#   核心：集成模型包装器 (Ensemble Wrapper)
# -------------------------------------------------
class EnsembleModel(torch.nn.Module):
    def __init__(self, config, model_paths, device):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.device = device
        
        logging.info(f"🌟 初始化集成模型，共有 {len(model_paths)} 个成员:")
        
        for path in model_paths:
            if not os.path.exists(path):
                logging.warning(f"⚠️ 权重文件不存在，跳过: {path}")
                continue
                
            # 构建单体模型结构
            model = build_model(config).to(device)
            
            # 加载权重
            try:
                checkpoint = torch.load(path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.models.append(model)
                logging.info(f"  ✅ 已加载: {os.path.basename(path)}")
            except Exception as e:
                logging.error(f"  ❌ 加载失败 {path}: {e}")
            
        if len(self.models) == 0:
            raise RuntimeError("❌ 没有加载到任何有效模型！请检查路径。")

    def forward(self, img, sig):
        # 收集所有模型的输出
        outputs = []
        for model in self.models:
            out = model(img, sig)
            outputs.append(out)
        
        # --- 核心集成策略：软投票 (Soft Voting) ---
        # 1. 对每个模型的输出做 Softmax，得到概率分布
        probs = [torch.softmax(out, dim=1) for out in outputs]
        
        # 2. 对概率取平均
        avg_probs = torch.mean(torch.stack(probs), dim=0)
        
        # 3. 返回 Logits (取 Log 保持数值特性兼容)
        return torch.log(avg_probs + 1e-8) 

# -------------------------------------------------
#   主程序
# -------------------------------------------------
def main():
    # 1. 配置加载
    config_path = 'config.json'
    if not os.path.exists(config_path):
        logging.error("未找到 config.json 文件")
        return

    # 【修复 1】强制使用 utf-8 读取配置，解决 GBK 报错
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # ================= 【修复 2】对接 GUI 配置 =================
    # 优先从 config['inference']['ensemble_models'] 读取列表
    # 如果该列表为空，则回退到 best_model_name (单模型模式)
    
    ensemble_weights = config.get('inference', {}).get('ensemble_models', [])
    
    # 过滤掉空字符串或无效路径
    ensemble_weights = [p for p in ensemble_weights if p and os.path.exists(p)]
    
    if not ensemble_weights:
        logging.warning("Config 中未找到有效的集成模型列表，回退到默认最佳模型 (Single Model Mode)。")
        default_path = os.path.join(config['train']['model_dir'], config['train']['best_model_name'])
        ensemble_weights = [default_path]
    else:
        logging.info(f"从配置中读取到 {len(ensemble_weights)} 个集成模型路径。")
    # ========================================================
    
    # 3. 初始化集成模型
    try:
        model = EnsembleModel(config, ensemble_weights, device)
    except Exception as e:
        logging.error(str(e))
        return

    # 4. 加载测试数据
    logging.info("Initializing Data Loaders...")
    # 只加载测试集
    _, _, test_loaders = create_dataloaders(config)
    
    if not test_loaders:
        logging.error("测试集加载失败，请检查 config 中的数据路径。")
        return

    # 5. 推理评估
    all_preds = []
    all_labels = []
    
    logging.info("Starting Ensemble Inference on Test Split...")
    
    with torch.no_grad():
        # 遍历所有测试 DataLoader
        for name, loader in test_loaders.items():
            for batch in tqdm(loader, desc=f"Testing ({name})"):
                # 解包数据 (兼容 dataset.py 的返回格式)
                if len(batch) == 3:
                    img, sig, lbl = batch
                elif len(batch) == 4:
                    img, sig, lbl, _ = batch
                else:
                    continue

                if img is None: continue
                
                img = img.to(device)
                sig = sig.to(device)
                lbl = lbl.to(device)
                
                # 集成模型前向传播
                output = model(img, sig)
                
                # 获取预测类别
                _, pred = torch.max(output, 1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())

    # 6. 生成报告
    acc = accuracy_score(all_labels, all_preds)
    logging.info("="*60)
    logging.info(f"🔥 FINAL RESULT ACCURACY: {acc:.4f} (Models: {len(ensemble_weights)})")
    logging.info("="*60)
    
    # 获取类别名称
    target_names = config['data'].get('case_ids', [str(i) for i in range(config['data']['num_classes'])])
    
    # 打印详细分类报告
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
    print("\nDetailed Classification Report:\n")
    print(report)
    
    # 7. 绘制并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix (Acc: {acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    log_dir = config['train']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # 注意：文件名必须与 Form1.cs 中的 logImgName 保持一致
    save_path = os.path.join(log_dir, 'final_generalization_matrix.png')
    plt.savefig(save_path)
    logging.info(f"Confusion Matrix saved to: {save_path}")
    
    # 保存详细文本报告
    with open(os.path.join(log_dir, 'ensemble_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Ensemble Accuracy: {acc:.4f}\n")
        f.write("Models used:\n")
        for p in ensemble_weights:
            f.write(f"- {p}\n")
        f.write("\n")
        f.write(report)

if __name__ == '__main__':
    main()