import os
import subprocess
import datetime
import re

# ================= 配置区域 =================
# 实验文件夹列表
EXP_IDS = ['222']

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
GENERALIZATION_SCRIPT = "generalization.py" # 假设每个子目录里都有这个文件
# ===========================================

def log_message(message, filepath):
    print(message)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def main():
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n{'='*50}\n🚀 批量抗干扰测试开始于: {start_time}\n{'='*50}"
    log_message(header, OUTPUT_FILE)
    
    log_message(f"{'Exp_ID':<10} | {'Condition':<12} | {'Accuracy':<10}", OUTPUT_FILE)
    log_message("-" * 40, OUTPUT_FILE)

    base_path = os.getcwd() # 记录根目录

    for exp_id in EXP_IDS:
        exp_dir = os.path.join(base_path, exp_id)
        config_path = os.path.join(exp_dir, "config.json")
        
        if not os.path.exists(config_path):
            log_message(f"{exp_id:<10} | SKIPPED (No config found)", OUTPUT_FILE)
            continue

        log_message(f"\n▶ 正在处理实验: {exp_id}", OUTPUT_FILE)

        # -------------------------------------------------
        # 关键修改：检查子目录里有没有 generalization.py
        # 如果没有，从根目录复制一个进去，确保能跑
        # -------------------------------------------------
        target_script = os.path.join(exp_dir, "generalization.py")
        if not os.path.exists(target_script):
            subprocess.run(f"cp generalization.py {target_script}", shell=True)
            subprocess.run(f"cp dataset.py {os.path.join(exp_dir, 'dataset.py')}", shell=True)
            subprocess.run(f"cp model.py {os.path.join(exp_dir, 'model.py')}", shell=True)

        for scenario_name, snr_val in SCENARIOS.items():
            env = os.environ.copy()
            if snr_val is not None:
                env["FORCE_SNR"] = snr_val
            elif "FORCE_SNR" in env:
                del env["FORCE_SNR"]
            
            try:
                # -------------------------------------------------
                # 核心修改：cwd=exp_dir
                # 让子进程在实验文件夹内部运行，这样相对路径 './checkpoints' 就是对的
                # -------------------------------------------------
                result = subprocess.run(
                    ['python', 'generalization.py'],
                    cwd=exp_dir,  # <--- 关键！切换工作目录
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                match = re.search(r"FINAL RESULT ACCURACY: (\d+\.\d+)", result.stdout) or \
                        re.search(r"FINAL RESULT ACCURACY: (\d+\.\d+)", result.stderr)
                
                if match:
                    acc_val = float(match.group(1)) * 100
                    acc_str = f"{acc_val:.2f}%"
                    print(f"   ✅ {scenario_name}: {acc_str}")
                    # 写入日志
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{exp_id},{scenario_name},{acc_str}\n")
                else:
                    print(f"   ❌ {scenario_name}: 失败")
                    # 打印错误详情
                    if result.stderr:
                        print(f"      [Error]: {result.stderr.splitlines()[-1]}")
            
            except Exception as e:
                print(f"系统错误: {e}")

    log_message(f"\n{'='*50}\n✅ 所有实验结束. 结果已保存至 {OUTPUT_FILE}\n{'='*50}", OUTPUT_FILE)

if __name__ == "__main__":
    main()