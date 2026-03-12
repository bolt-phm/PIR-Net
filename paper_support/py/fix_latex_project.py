import os
import re
import shutil

def fix_latex_project():
    target_file = "luoshuan.tex"
    backup_file = "luoshuan_backup.tex"
    
    print(f"🚀 开始修复: {target_file}")
    
    if not os.path.exists(target_file):
        print(f"❌ 错误: 找不到文件 {target_file}")
        return

    # 1. 自动备份 (防止意外)
    shutil.copy(target_file, backup_file)
    print(f"📦 已创建备份: {backup_file}")

    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # =========================================================
    # 修复 A: 彻底清除所有 和 [cite_end] 标记
    # 这是导致 "Missing $ inserted" 的罪魁祸首
    # =========================================================
    # 使用正则替换，忽略大小写，确保清理干净
    content = re.sub(r'\[cite_start\]', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\[cite_end\]', '', content, flags=re.IGNORECASE)
    print("✅ 已彻底清除所有 /[cite_end] 标记")

    # =========================================================
    # 修复 B: 修复特殊破折号 (Missing character 来源)
    # =========================================================
    if "—" in content:
        content = content.replace("—", "---")
        print("✅ 已修复非法中文破折号")

    # =========================================================
    # 修复 C: 修复图片过宽 (Float too large 来源)
    # =========================================================
    # 将 width=1.0\linewidth 替换为 0.9\linewidth
    content = re.sub(r'(width\s*=\s*)(1\.0\\linewidth|\\linewidth)', r'\g<1>0.9\\linewidth', content)
    print("✅ 已自动调整图片宽度")

    # =========================================================
    # 修复 D: 更新实验数据 (修正 Python 警告写法)
    # =========================================================
    # 使用 raw string (r"") 避免 invalid escape sequence 警告
    replacements = {
        r"96.04%": r"95.00%",
        r"96.04\%": r"95.00\%",
        r"98.62%": r"98.00%",
        r"98.62\%": r"98.00\%",
        r"zero false negatives": r"minimizing false alarms",
        r"99.86%": r"99.80%",
        r"99.86\%": r"99.80\%"
    }
    
    count = 0
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            count += 1
    print(f"✅ 已同步更新 {count} 处实验数据")

    # =========================================================
    # 3. 直接覆盖写入原文件
    # =========================================================
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n" + "="*40)
    print("🎉 修复成功！原文件已被覆盖更新。")
    print("👉 现在请直接运行您的编译脚本/命令即可！")
    print("="*40)

if __name__ == "__main__":
    fix_latex_project()