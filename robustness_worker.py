import os
import json
import torch
import logging
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

# 引入自定义模块
# 确保 dataset.py 和 model.py 在当前目录下
from dataset import create_dataloaders
from model import build_model 

# 配置日志：只显示错误级别，保持控制台输出干净，方便主程序解析
logging.basicConfig(level=logging.ERROR)

# =================================================
#  集成模型包装器 (Ensemble Model Wrapper)
# =================================================
class EnsembleModel(torch.nn.Module):
    def __init__(self, config, model_paths, device):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.device = device
        
        # 加载所有子模型
        valid_models = 0
        for path in model_paths:
            if not os.path.exists(path):
                continue
                
            try:
                # 构建模型结构
                model = build_model(config).to(device)
                
                # 加载权重
                checkpoint = torch.load(path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.models.append(model)
                valid_models += 1
            except Exception as e:
                # 仅在加载失败时打印错误，不中断程序
                pass
            
        if valid_models == 0:
            raise RuntimeError("CRITICAL: 没有加载到任何有效模型，无法进行测试！")

    def forward(self, img, sig):
        # 收集所有模型的输出
        outputs = []
        for model in self.models:
            out = model(img, sig)
            outputs.append(out)
        
        # 软投票 (Soft Voting): 先 Softmax 归一化，再取平均
        probs = [torch.softmax(out, dim=1) for out in outputs]
        avg_probs = torch.mean(torch.stack(probs), dim=0)
        
        # 返回 Logits 形式 (取 log)
        return torch.log(avg_probs + 1e-8) 

# =================================================
#  主逻辑
# =================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to experiment config.json')
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config_path):
        print("ERROR_CONFIG_NOT_FOUND")
        return

    # 获取 config 文件所在的目录 (例如 "022")，用于处理相对路径
    config_dir = os.path.dirname(args.config_path)

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')

    # 2. 处理模型路径 (智能修复相对路径)
    # 优先使用 config['inference']['ensemble_models']
    ensemble_weights = config.get('inference', {}).get('ensemble_models', [])
    
    # 如果列表为空，回退到 best_model
    if not ensemble_weights:
        model_name = config['train']['best_model_name']
        model_dir = config['train']['model_dir']
        # 拼接路径
        ensemble_weights = [os.path.join(model_dir, model_name)]

    # 修正路径：如果直接路径找不到，尝试拼接到 config_dir 下
    fixed_paths = []
    for p in ensemble_weights:
        if os.path.exists(p):
            fixed_paths.append(p)
        else:
            # 尝试拼接子文件夹路径
            rel_path = os.path.join(config_dir, p)
            if os.path.exists(rel_path):
                fixed_paths.append(rel_path)
    
    if not fixed_paths:
        print("ERROR_NO_MODELS_FOUND")
        return

    # 3. 初始化集成模型
    try:
        model = EnsembleModel(config, fixed_paths, device)
    except Exception:
        print("ERROR_MODEL_INIT_FAILED")
        return

    # 4. 加载数据
    # dataset.py 会自动读取环境变量 FORCE_SNR 并在内部处理噪声
    # 这里我们不需要做任何额外操作
    try:
        # 假设我们只关心 'test' 集，或者根据 config 决定
        _, _, test_loaders = create_dataloaders(config)
    except Exception:
        print("ERROR_DATA_LOAD_FAILED")
        return

    if not test_loaders:
        print("ERROR_NO_TEST_DATA")
        return

    # 5. 推理循环
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 遍历所有测试 Loader (通常只有一个，但为了兼容字典返回格式)
        for name, loader in test_loaders.items():
            for batch in loader:
                # 解包数据
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

                # =================================================
                # 注意：此处不再手动注入噪声！
                # 噪声已由 Dataset 在生成阶段根据环境变量注入
                # img 和 sig 已经是"脏"的且物理特征对齐的
                # =================================================

                output = model(img, sig)
                _, pred = torch.max(output, 1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())

    # 6. 计算并输出结果
    if len(all_labels) == 0:
        print("ERROR_NO_SAMPLES")
        return

    acc = accuracy_score(all_labels, all_preds)
    
    # 输出特定格式供主程序 (run_batch_kgr.py) 解析
    print(f"FINAL_ACCURACY_RESULT:{acc*100:.2f}")

if __name__ == '__main__':
    main()