#!/bin/bash

# 获取当前脚本所在目录作为基础路径
BASE_PATH=$(pwd)

# =========================================================
# 1. 修正数组定义 (每个元素分开写，或者直接用空格隔开)
# =========================================================
EXPERIMENTS=("222")

echo "=========================================="
echo "      自动化训练脚本启动 (切换目录模式)"
echo "      总计实验数: ${#EXPERIMENTS[@]}"
echo "=========================================="

for exp in "${EXPERIMENTS[@]}"; do
    target_dir="$BASE_PATH/$exp"
    
    # 定义该实验内部的日志存放路径 (相对路径)
    log_folder="logs/running_log"
    log_file="$log_folder/running_log.txt"
    
    echo ""
    echo ">>>>>> [$(date)] 准备运行实验: $exp <<<<<<"
    echo "工作目录: $target_dir"
    
    if [ -d "$target_dir" ]; then
        
        # =========================================================
        # 2. 这里的 cp 命令非常重要！
        # 如果子文件夹里没有 train.py，必须把根目录的复制进去
        # =========================================================
        echo "正在同步代码 (train.py, model.py, dataset.py) 到 $target_dir ..."
        cp "$BASE_PATH/train.py" "$target_dir/"
        cp "$BASE_PATH/model.py" "$target_dir/"
        cp "$BASE_PATH/dataset.py" "$target_dir/"
        # cp "$BASE_PATH/utils.py" "$target_dir/"  # 如果有其他文件自行添加
        
        # 切换工作目录
        cd "$target_dir" || exit
        
        # 创建日志目录
        if [ ! -d "$log_folder" ]; then
            mkdir -p "$log_folder"
        fi
        
        echo "正在启动训练... (已进入目录 $exp)"
        echo "日志输出: $exp/$log_file"
        
        # 运行训练
        python -u train.py > "$log_file" 2>&1
        
        # 检查结果
        if [ $? -eq 0 ]; then
            echo ">>>>>> [$(date)] 实验 $exp 运行成功！"
        else
            echo "!!!!!! [$(date)] 实验 $exp 运行异常，请查看日志。"
        fi
        
        # 切回根目录
        cd "$BASE_PATH" || exit
        
    else
        echo "!!!!!! 错误: 找不到目录 $target_dir，跳过..."
    fi
    
    # 休息 5 秒
    sleep 5
done

echo ""
echo "=========================================="
echo "      所有实验计划执行完毕"
echo "=========================================="