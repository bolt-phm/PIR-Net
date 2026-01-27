import os
import subprocess
import datetime
import re

# ================= 配置区域 =================
# 实验文件夹列表 (根据你的截图)
EXP_IDS = ['022', '122', '202', '212', '220', '221', '222']

# 测试的 SNR 档位
SCENARIOS = {
    "Clean": None,
    "+20dB": "20",
    "+10dB": "10",
    "+5dB": "5",
    "0dB": "0",
    "-5dB": "-5"
}

OUTPUT_FILE = "kgr.log"
WORKER_SCRIPT = "robustness_worker.py"
# ===========================================

def log_message(message, filepath):
    """同时打印到控制台并写入文件"""
    print(message)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def main():
    # 初始化日志文件头
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'='*50}\n🚀 批量抗干扰测试开始于: {start_time}\n{'='*50}"
    log_message(header, OUTPUT_FILE)
    
    # 表头
    log_message(f"{'Exp_ID':<10} | {'Condition':<12} | {'Accuracy':<10}", OUTPUT_FILE)
    log_message("-" * 40, OUTPUT_FILE)

    for exp_id in EXP_IDS:
        config_path = os.path.join(exp_id, "config.json")
        
        # 检查配置是否存在
        if not os.path.exists(config_path):
            log_message(f"{exp_id:<10} | SKIPPED (No config found)", OUTPUT_FILE)
            continue

        log_message(f"\n▶ 正在处理实验: {exp_id}", OUTPUT_FILE)

        for scenario_name, snr_val in SCENARIOS.items():
            # 准备环境变量
            env = os.environ.copy()
            if snr_val is not None:
                env["FORCE_SNR"] = snr_val
                # 清理之前的可能残留 (虽不是必须，但保险)
            else:
                if "FORCE_SNR" in env:
                    del env["FORCE_SNR"]
            
            # 调用 worker
            # 这里的命令相当于: python robustness_worker.py --config_path 222/config.json
            try:
                result = subprocess.run(
                    ['python', WORKER_SCRIPT, '--config_path', config_path],
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                # 解析输出
                # 我们在 worker 里 print 了 "FINAL_ACCURACY_RESULT:xx.xx"
                match = re.search(r"FINAL_ACCURACY_RESULT:(\d+\.\d+)", result.stdout)
                
                if match:
                    acc = match.group(1)
                    log_line = f"{exp_id:<10} | {scenario_name:<12} | {acc}%"
                    # 这里为了美观，我们在控制台打印简略版，但写入 kgr.log 的是标准格式
                    print(f"   ✅ {scenario_name}: {acc}%")
                    # 将这一行追加到 kgr.log (使用 append 模式)
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{exp_id},{scenario_name},{acc}\n") # 建议保存为 CSV 格式方便后续分析，或者保持上面的竖线格式
                else:
                    error_msg = f"{exp_id:<10} | {scenario_name:<12} | ERROR"
                    print(f"   ❌ {scenario_name}: 失败")
                    # 打印一点错误日志方便调试
                    if result.stderr:
                        print(f"      [Error Detail]: {result.stderr.splitlines()[-1]}")
                    elif result.stdout:
                         print(f"      [Error Detail]: {result.stdout.splitlines()[-1]}")
            
            except Exception as e:
                print(f"系统错误: {e}")

    log_message(f"\n{'='*50}\n✅ 所有实验结束. 结果已保存至 {OUTPUT_FILE}\n{'='*50}", OUTPUT_FILE)

if __name__ == "__main__":
    main()