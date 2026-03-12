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
from model import build_model 

# 强制设置日志编码
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =================================================
#  工具函数：强制噪声注入
# =================================================
def inject_awgn_tensor(inputs, snr_db, description="Tensor"):
    """
    inputs: Tensor
    snr_db: float
    description: 用于打印调试信息的名字
    return: noisy_inputs, debug_info_str
    """
    if snr_db is None:
        return inputs, None
    
    inputs = inputs.float()
    
    # 计算信号平均功率
    flat_inputs = inputs.view(inputs.size(0), -1)
    signal_power = torch.mean(flat_inputs ** 2, dim=1, keepdim=True) # [N, 1]
    avg_signal_power = torch.mean(signal_power).item() # 仅用于打印
    
    # 计算噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    avg_noise_power = torch.mean(noise_power).item() # 仅用于打印
    
    # 生成噪声
    noise = torch.randn_like(inputs)
    
    # 调整幅度
    view_shape = [inputs.size(0)] + [1] * (inputs.ndim - 1)
    scale = torch.sqrt(noise_power).view(*view_shape)
    
    noisy_input = inputs + noise * scale
    
    debug_msg = f"[{description}] P_sig:{avg_signal_power:.5f} | P_noise:{avg_noise_power:.5f} (SNR={snr_db})"
    return noisy_input, debug_msg

# -------------------------------------------------
#   集成模型包装器 (保持不变)
# -------------------------------------------------
class EnsembleModel(torch.nn.Module):
    def __init__(self, config, model_paths, device):
        super().__init__()
        self.models = torch.nn.ModuleList()
        self.device = device
        logging.info(f"🌟 初始化集成模型，共有 {len(model_paths)} 个成员:")
        for path in model_paths:
            if not os.path.exists(path):
                continue
            model = build_model(config).to(device)
            try:
                checkpoint = torch.load(path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                self.models.append(model)
            except Exception as e:
                logging.error(f"  ❌ 加载失败 {path}: {e}")
            
        if len(self.models) == 0:
            raise RuntimeError("❌ 没有加载到任何有效模型！")

    def forward(self, img, sig):
        outputs = []
        for model in self.models:
            out = model(img, sig)
            outputs.append(out)
        probs = [torch.softmax(out, dim=1) for out in outputs]
        avg_probs = torch.mean(torch.stack(probs), dim=0)
        return torch.log(avg_probs + 1e-8) 

# -------------------------------------------------
#   主程序
# -------------------------------------------------
def main():
    config_path = 'config.json'
    if not os.path.exists(config_path):
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    
    # 模型加载逻辑
    ensemble_weights = config.get('inference', {}).get('ensemble_models', [])
    ensemble_weights = [p for p in ensemble_weights if p and os.path.exists(p)]
    if not ensemble_weights:
        default_path = os.path.join(config['train']['model_dir'], config['train']['best_model_name'])
        ensemble_weights = [default_path]
    
    try:
        model = EnsembleModel(config, ensemble_weights, device)
    except Exception as e:
        logging.error(str(e))
        return

    logging.info("Initializing Data Loaders...")
    _, _, test_loaders = create_dataloaders(config)
    
    if not test_loaders:
        return

    # ==== 获取强制 SNR 参数 ====
    forced_snr_str = os.environ.get('FORCE_SNR')
    forced_snr = float(forced_snr_str) if forced_snr_str is not None and forced_snr_str != "" else None
    
    if forced_snr is not None:
        logging.info(f"⚠️ [DEBUG] 强制噪声注入模式已激活! SNR = {forced_snr} dB")
        logging.info(f"⚠️ [DEBUG] 将同时攻击 Signal 和 Image 通道！")

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for name, loader in test_loaders.items():
            # 增加一个 batch 计数器，只在第一个 batch 打印调试信息
            batch_idx = 0
            for batch in tqdm(loader, desc=f"Testing ({name})"):
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
                
                # ===========================================
                # 🚀 强制注入噪声 (攻击所有输入)
                # ===========================================
                if forced_snr is not None:
                    # 1. 攻击 Signal
                    sig, debug_sig = inject_awgn_tensor(sig, forced_snr, "Signal")
                    # 2. 攻击 Image (假设 img 也是浮点数，如果是频谱图，这相当于加噪)
                    img, debug_img = inject_awgn_tensor(img, forced_snr, "Image")
                    
                    # 只打印第一个 batch 的能量统计，证明我们真的改了数据
                    if batch_idx == 0:
                        print(f"\n[DEBUG CHECK] {debug_sig}")
                        print(f"[DEBUG CHECK] {debug_img}")
                        # 再次确认数据范围
                        print(f"[DEBUG CHECK] Sig range: {sig.min():.2f} to {sig.max():.2f}")
                        print(f"[DEBUG CHECK] Img range: {img.min():.2f} to {img.max():.2f}\n")
                
                batch_idx += 1
                # ===========================================
                
                output = model(img, sig)
                _, pred = torch.max(output, 1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    mode_str = f"SNR={forced_snr}dB" if forced_snr is not None else "Clean"
    logging.info(f"🔥 FINAL RESULT ACCURACY: {acc:.4f} [{mode_str}]")
    
    target_names = config['data'].get('case_ids', [str(i) for i in range(config['data']['num_classes'])])
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)
    
    # 打印结果（保留原有逻辑）
    print(f"FINAL RESULT ACCURACY: {acc:.4f}") # 供 Launcher 解析
    
    # 保存结果
    log_dir = config['train']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    filename_suffix = f"_SNR{int(forced_snr)}" if forced_snr is not None else ""
    
    with open(os.path.join(log_dir, f'ensemble_report{filename_suffix}.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == '__main__':
    main()