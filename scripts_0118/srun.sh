#!/bin/bash

# 检查是否提供了脚本路径作为参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <script_path>"
    exit 1
fi

# 获取传入的脚本路径
SCRIPT_PATH=$1

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script $SCRIPT_PATH not found!"
    exit 1
fi

# 获取脚本名称（去掉路径和扩展名）
SCRIPT_NAME=$(basename "$SCRIPT_PATH" .sh)

# 创建输出目录，目录名称为脚本名称
OUTPUT_DIR="/mnt/petrelfs/mengfanqing/Codebook/slurm/$SCRIPT_NAME"
mkdir -p $OUTPUT_DIR

# 获取当前时间戳（格式为：YYYY-MM-DD_HH-MM-SS）
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
# 设置作业输出文件路径，文件名中包含当前时间
SBATCH_OUTPUT_PATH="$OUTPUT_DIR/phoenix-slurm-$CURRENT_TIME.out"


# 运行脚本并将输出重定向到指定目录
echo "Running script $SCRIPT_PATH..."
srun -p  gvembodied -c 48 --gres=gpu:8 --async --quotatype=reserved --output=$SBATCH_OUTPUT_PATH \
    sh $SCRIPT_PATH

# 提示任务已启动 Gveval-S1 gvembodied
echo "Job started. Output and batch script will be saved to $OUTPUT_DIR"
