#!/bin/bash
set -e -x

CKPT_PATH=$1
CKPT_NAME=$2
CONV_MODE=${3:-qwen_2}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$CKPT_NAME
ROOT_DIR="/blob/v-jiaqiliao/Results/Codebook/eval/scienceqa"

# 创建日志目录和主日志文件
log_file="$ROOT_DIR/answers/$CKPT/eval.log"
mkdir -p $(dirname $log_file)
> "$log_file"

echo "Starting ScienceQA evaluation with $CHUNKS GPU(s)" > "$log_file"
echo "Using GPU(s): $gpu_list" >> "$log_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    # 为每个GPU创建单独的日志文件
    gpu_log_file="$ROOT_DIR/answers/$CKPT/gpu_${GPULIST[$IDX]}.log"
    > "$gpu_log_file"
    echo "Starting evaluation on GPU ${GPULIST[$IDX]} (Chunk $((IDX+1))/$CHUNKS)" >> "$gpu_log_file"
    
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-path $CKPT_PATH \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/test \
        --answers-file $ROOT_DIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode $CONV_MODE >> "$gpu_log_file" 2>&1 &
done

echo "Waiting for all evaluation processes to complete..." >> "$log_file"
wait

output_file="$ROOT_DIR/answers/$CKPT/merge.jsonl"
echo "Merging results into $output_file" >> "$log_file"

# 清除输出文件
> "$output_file"

# 合并所有分片结果
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

echo "Evaluating final results..." >> "$log_file"
python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa/ \
    --result-file $output_file \
    --output-file $ROOT_DIR/answers/$CKPT/output.jsonl \
    --output-result $ROOT_DIR/answers/$CKPT/result.json >> "$log_file" 2>&1
