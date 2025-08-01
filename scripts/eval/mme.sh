#!/bin/bash
set -e -x

CKPT_PATH=$1
CKPT_NAME=$2
CONV_MODE=$3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$CKPT_NAME
ROOT_DIR="/mnt/data/jiaqi.liao/Codebook/eval_results/mme"

# Create log file path and clear old content
log_file="$ROOT_DIR/answers/$CKPT/eval.log"
mkdir -p $(dirname $log_file)
> "$log_file"

echo "Starting MME evaluation with $CHUNKS GPU(s)" > "$log_file"
echo "Using GPU(s): $gpu_list" >> "$log_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    # Create separate log file for each GPU and clear old content
    gpu_log_file="$ROOT_DIR/answers/$CKPT/gpu_${GPULIST[$IDX]}.log"
    > "$gpu_log_file"
    echo "Starting evaluation on GPU ${GPULIST[$IDX]} (Chunk $((IDX+1))/$CHUNKS)" >> "$gpu_log_file"
    
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT_PATH \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
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

cp $ROOT_DIR/answers/$CKPT/merge.jsonl ./playground/data/eval/MME/answers/$CKPT.jsonl

cd ./playground/data/eval/MME

echo "Converting answers to MME format..." >> "$log_file"
python convert_answer_to_mme.py --experiment $CKPT >> "$log_file" 2>&1

cd eval_tool

echo "Calculating MME metrics..." >> "$log_file"
python calculation.py --results_dir answers/$CKPT >> "$log_file" 2>&1

#!/bin/bash