#!/bin/bash

# 定义基础路径
BASE_PATH="/root/autodl-tmp/fanhu"

# 定义你要跑的实验文件夹列表
EXPERIMENTS=( "212")

echo "=========================================="
echo "      自动化训练脚本启动 (独立日志模式)"
echo "=========================================="

for exp in "${EXPERIMENTS[@]}"; do
    target_dir="$BASE_PATH/$exp"
    
    # 定义该实验内部的日志存放路径 (相对路径)
    # 根据你的要求: ./logs/running_log/running_log.txt
    log_folder="logs/running_log"
    log_file="$log_folder/running_log.txt"
    
    echo ""
    echo ">>>>>> [$(date)] 准备运行实验: $exp <<<<<<"
    echo "工作目录: $target_dir"
    
    if [ -d "$target_dir" ]; then
        cd "$target_dir"
        
        # 1. 自动创建日志文件夹 (如果不存在的话)
        if [ ! -d "$log_folder" ]; then
            echo "创建日志目录: $log_folder"
            mkdir -p "$log_folder"
        fi
        
        echo "正在启动训练... 控制台输出将保存至: $target_dir/$log_file"
        
        # 2. 运行训练代码并重定向输出
        # > "$log_file"  : 将标准输出(stdout)覆盖写入到文件
        # 2>&1           : 将错误输出(stderr)也重定向到标准输出，一并写入文件
        python -u train.py > "$log_file" 2>&1
        
        # 3. 检查运行结果
        if [ $? -eq 0 ]; then
            echo ">>>>>> [$(date)] 实验 $exp 运行成功完成！"
        else
            echo "!!!!!! [$(date)] 实验 $exp 运行异常，请查看 $log_file 获取报错详情。"
        fi
        
    else
        echo "!!!!!! 错误: 找不到目录 $target_dir，跳过..."
    fi
    
    # 休息 5 秒，释放显存
    sleep 5
done

echo ""
echo "=========================================="
echo "      所有实验计划执行完毕"
echo "=========================================="