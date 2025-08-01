#!/bin/bash
set -e -x

CKPT_PATH=$1
CKPT_NAME=$2
CONV_MODE=$3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
SPLIT="mmbench_dev_20230712"
ROOT_DIR="/mnt/data/jiaqi.liao/Codebook/eval_results/mmbench"

# Create log file path and clear old content
log_file="$ROOT_DIR/answers/$CKPT_NAME/eval.log"
mkdir -p $(dirname $log_file)
> "$log_file"

echo "Starting MMBench evaluation with $CHUNKS GPU(s)" > "$log_file"
echo "Using GPU(s): $gpu_list" >> "$log_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    # Create separate log file for each GPU and clear old content
    gpu_log_file="$ROOT_DIR/answers/$CKPT_NAME/gpu_${GPULIST[$IDX]}.log"
    > "$gpu_log_file"
    echo "Starting evaluation on GPU ${GPULIST[$IDX]} (Chunk $((IDX+1))/$CHUNKS)" >> "$gpu_log_file"
    
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_mmbench \
        --model-path $CKPT_PATH \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file $ROOT_DIR/answers/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode $CONV_MODE \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX >> "$gpu_log_file" 2>&1 &
done

echo "Waiting for all evaluation processes to complete..." >> "$log_file"
wait

output_file="$ROOT_DIR/answers/$CKPT_NAME/$CKPT_NAME.jsonl"
echo "Merging results into $output_file" >> "$log_file"

# Clear output file (if exists)
> "$output_file"

# Merge all chunk results
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DIR/answers/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $ROOT_DIR/answers_upload/$CKPT_NAME

echo "Converting results for submission..." >> "$log_file"
python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir $ROOT_DIR/answers/$CKPT_NAME \
    --upload-dir $ROOT_DIR/answers_upload/$CKPT_NAME \
    --experiment $CKPT_NAME >> "$log_file" 2>&1
