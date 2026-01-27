import os
import json
import subprocess
import re
import pandas as pd
import time

# 配置文件路径（仅用于备份，防止意外损坏，实际逻辑主要依赖环境变量）
CONFIG_PATH = 'config.json'
BACKUP_PATH = 'config.json.bak'

# =========================================================
# 定义测试场景
# env_snr: 对应 generalization.py 中读取的 'FORCE_SNR' 环境变量
# =========================================================
SCENARIOS = {
    "Clean (Baseline)": {
        "env_snr": None 
    },
    "+20dB AWGN": {
        "env_snr": "20"
    },
    "+10dB AWGN": {
        "env_snr": "10"
    },
    "+5dB AWGN": {
        "env_snr": "5"
    },
    "0dB AWGN": {
        "env_snr": "0"
    },
    "-5dB AWGN": {
        "env_snr": "-5"
    }
}

def parse_accuracy(output_log_stderr, output_log_stdout):
    """
    从日志中提取准确率。优先检查 stderr (因为 logging 默认输出到 stderr)
    """
    # 匹配 generalization.py 中打印的格式: "FINAL RESULT ACCURACY: 0.xxxx"
    pattern = r"FINAL RESULT ACCURACY: (\d+\.\d+)"
    
    # 1. 尝试从 stderr 找
    match = re.search(pattern, output_log_stderr)
    if match:
        return float(match.group(1)) * 100
        
    # 2. 尝试从 stdout 找
    match = re.search(pattern, output_log_stdout)
    if match:
        return float(match.group(1)) * 100
        
    return 0.0

def main():
    # 1. 安全备份配置文件
    if not os.path.exists(BACKUP_PATH) and os.path.exists(CONFIG_PATH):
        subprocess.run(f"cp {CONFIG_PATH} {BACKUP_PATH}", shell=True)
    
    results = []
    print(f"🚀 开始执行抗干扰基准测试 (强制注入模式 - 共 {len(SCENARIOS)} 项)...\n")
    
    try:
        for name, cfg in SCENARIOS.items():
            print(f"▶ 正在运行: {name}")
            
            # 2. 准备环境变量
            # 复制当前系统的环境变量，避免丢失 PATH 等重要信息
            my_env = os.environ.copy()
            
            # 设置强制 SNR 参数
            target_snr = cfg.get("env_snr")
            if target_snr is not None:
                my_env["FORCE_SNR"] = target_snr
                print(f"   [配置] 设置环境变量 FORCE_SNR = {target_snr}")
            else:
                # 显式删除，防止上一轮循环的残留
                if "FORCE_SNR" in my_env:
                    del my_env["FORCE_SNR"]
                print(f"   [配置] 标准模式 (无强制噪声)")

            # 3. 启动子进程进行测试
            # 注意：这里不再修改 config.json，而是依赖环境变量传参
            start_time = time.time()
            result = subprocess.run(
                ['python', 'generalization.py'], 
                capture_output=True, 
                text=True,
                env=my_env  # <--- 关键：将环境变量传入子进程
            )
            elapsed = time.time() - start_time
            
            # 4. 解析结果
            acc = parse_accuracy(result.stderr, result.stdout)
            
            # 检查是否有特定的 Debug 标记，确认注入是否成功
            if target_snr is not None and "强制噪声注入模式已激活" in result.stderr:
                print("   [Check] ✅ 成功检测到注入日志")
            
            if acc > 0:
                print(f"   ✅ 完成! Accuracy: {acc:.2f}% (耗时: {elapsed:.1f}s)\n")
            else:
                print(f"   ❌ 失败或未找到结果. (耗时: {elapsed:.1f}s)")
                # 如果失败，打印最后几行日志以便调试
                print("   [Error Log Tail]:")
                print("\n".join(result.stderr.splitlines()[-5:]))
                print("\n")

            results.append({
                "Condition": name, 
                "SNR Setting": target_snr if target_snr else "None",
                "Accuracy (%)": f"{acc:.2f}"
            })
            
    finally:
        # 恢复原始配置 (虽然本次未修改 config，但保持习惯是个好事)
        if os.path.exists(BACKUP_PATH):
            print("🔄 检查环境一致性...")
            # subprocess.run(f"cp {BACKUP_PATH} {CONFIG_PATH}", shell=True)
    
    # 5. 输出最终报告
    print("\n" + "="*50)
    print("       ROBUSTNESS EVALUATION REPORT       ")
    print("="*50)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 保存 CSV
    csv_path = "robustness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📄 结果已保存至: {csv_path}")

if __name__ == "__main__":
    main()