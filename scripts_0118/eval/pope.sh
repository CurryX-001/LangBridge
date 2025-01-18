#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 获取模型路径（CKPT_PATH）和 conv_mode 参数
CKPT_PATH=${1}
CONV_MODE=${2:-llama3}  # 如果没有传入 conv_mode，默认使用 llama3

# 获取 CKPT 文件名（模型名称）
CKPT=$(basename ${CKPT_PATH})

# 设置输出日志文件路径
CURRENT_TIME=$(date +'%Y-%m-%d_%H-%M-%S')
LOG_DIR="/mnt/petrelfs/mengfanqing/Codebook/eval_log/${CKPT}/pope/${CURRENT_TIME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/output.log"

# 记录日志的开头部分
echo "Starting evaluation for model: ${CKPT}" > ${LOG_FILE}
echo "Using conv_mode: ${CONV_MODE}" >> ${LOG_FILE}
echo "Timestamp: ${CURRENT_TIME}" >> ${LOG_FILE}

# 运行 model_vqa_loader 脚本
python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_PATH} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV_MODE} \
    >> ${LOG_FILE} 2>&1

# 运行 eval_pope.py 脚本
python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${CKPT}.jsonl \
    >> ${LOG_FILE} 2>&1

echo "POPE Evaluation finished for model: ${CKPT}" >> ${LOG_FILE}
