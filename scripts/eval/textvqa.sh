#!/bin/bash
set -e -x
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_PATH=$1
CKPT_NAME=$2
CONV_MODE=$3

CKPT=$CKPT_NAME
SPLIT="llava_textvqa_val_v051_ocr"
ROOT_DIR="/mnt/data/jiaqi.liao/Codebook/eval_results/textvqa"

# Create log file path and clear old content
log_file="$ROOT_DIR/answers/$CKPT/eval.log"
mkdir -p $(dirname $log_file)
> "$log_file"

echo "Starting TextVQA evaluation with $CHUNKS GPU(s)" > "$log_file"
echo "Using GPU(s): $gpu_list" >> "$log_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    # Create separate log file for each GPU and clear old content
    gpu_log_file="$ROOT_DIR/answers/$CKPT/gpu_${GPULIST[$IDX]}.log"
    > "$gpu_log_file"
    echo "Starting evaluation on GPU ${GPULIST[$IDX]} (Chunk $((IDX+1))/$CHUNKS)" >> "$gpu_log_file"
    
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file /mnt/data/jiaqi.liao/Codebook/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder /mnt/data/jiaqi.liao/Codebook/playground/data/eval/textvqa/train_images \
        --answers-file $ROOT_DIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE >> "$gpu_log_file" 2>&1 &
done

echo "Waiting for all evaluation processes to complete..." >> "$log_file"
wait

output_file="$ROOT_DIR/answers/$CKPT/merge.jsonl"
echo "Merging results into $output_file" >> "$log_file"

# Clear output file (if exists)
> "$output_file"

# Merge all chunk results
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ROOT_DIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

echo "Running TextVQA evaluation..." >> "$log_file"
python -m llava.eval.eval_textvqa \
    --annotation-file /mnt/data/jiaqi.liao/Codebook/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file >> "$log_file" 2>&1
