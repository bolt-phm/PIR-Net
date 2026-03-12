import torch
import time
import numpy as np
from model import build_model
import json
from thop import profile, clever_format

def main():
    # 1. 加载配置 (为了构建模型)
    config_path = '222/config.json'  # 任选一个存在的配置文件
    if not os.path.exists(config_path):
        print("请修改 config_path 为你实际存在的 json 路径")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 构建模型
    print("正在构建模型...")
    model = build_model(config).to(device)
    model.eval()

    # 3. 创建虚拟输入 (根据你的 dataset.py 输出形状)
    # Image: [Batch, 5, 224, 224] (假设 ResNet101 输入)
    # Signal: [Batch, 1, 8192]
    batch_size = 1
    dummy_img = torch.randn(batch_size, 5, 224, 224).to(device)
    dummy_sig = torch.randn(batch_size, 1, 8192).to(device)

    # 4. 计算 FLOPs 和 Params
    print("\n正在计算 FLOPs 和 参数量...")
    try:
        # thop 需要输入是 tuple
        flops, params = profile(model, inputs=(dummy_img, dummy_sig), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        print(f"===================================")
        print(f"FLOPs (计算量): {flops_str}")
        print(f"Params (参数量): {params_str}")
        print(f"===================================")
    except Exception as e:
        print(f"THOP 计算失败 (可能是模型结构特殊): {e}")

    # 5. 测试推理速度 (FPS)
    print("\n正在测试推理速度 (Warmup + 1000次循环)...")
    
    # 预热
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_img, dummy_sig)
            
    # 正式计时
    t_start = time.time()
    n_loops = 1000
    with torch.no_grad():
        for _ in range(n_loops):
            _ = model(dummy_img, dummy_sig)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize() # 等待 GPU以此确保计时准确
        
    t_end = time.time()
    
    total_time = t_end - t_start
    avg_time_ms = (total_time / n_loops) * 1000
    fps = n_loops / total_time
    
    print(f"Avg Inference Time: {avg_time_ms:.2f} ms")
    print(f"FPS (Throughput): {fps:.2f} samples/sec")
    print(f"===================================")

if __name__ == '__main__':
    import os
    main()