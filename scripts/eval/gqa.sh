#!/bin/bash
set -e -x

CKPT_PATH=$1
CKPT_NAME=$2
CONV_MODE=$3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$CKPT_NAME
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"
ROOT_DIR="/mnt/data/jiaqi.liao/Codebook/eval_results/gqa"

# 创建日志文件路径并清除旧内容
log_file="$ROOT_DIR/answers/$SPLIT/$CKPT/eval.log"
mkdir -p $(dirname $log_file)
> "$log_file"  # 清除日志文件内容

echo "Starting GQA evaluation with $CHUNKS GPU(s)" > "$log_file"
echo "Using GPU(s): $gpu_list" >> "$log_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    # 为每个GPU创建单独的日志文件并清除旧内容
    gpu_log_file="$ROOT_DIR/answers/$SPLIT/$CKPT/gpu_${GPULIST[$IDX]}.log"
    > "$gpu_log_file"  # 清除GPU特定的日志文件内容
    echo "Starting evaluation on GPU ${GPULIST[$IDX]} (Chunk $((IDX+1))/$CHUNKS)" >> "$gpu_log_file"
    
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images \
        --answers-file $ROOT_DIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE >> "$gpu_log_file" 2>&1 &
done

echo "Waiting for all evaluation processes to complete..." >> "$log_file"
wait

output_file=$ROOT_DIR/answers/$SPLIT/$CKPT/merge.jsonl
echo "Merging results into $output_file" >> "$log_file"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

echo "Converting results to GQA format..." >> "$log_file"
python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json >> "$log_file" 2>&1

echo "Running GQA evaluation..." >> "$log_file"
cd $GQADIR
python 1_eval.py --tier testdev_balanced >> "$log_file" 2>&1
