#!/bin/bash

CKPT_PATH_list=(
    "./checkpoints/llama3_it_8b_it_vocab_19200_dim_5120_finetune_mmvocab"
    "./checkpoints/qwen2_0.5b_it_vocab_19200_dim_5120_finetune_mmvocab"
)

conv_mode_list=(
    "llama3"
    "qwen_2"  
)

eval_dataset_list=(
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
)

for i in "${!CKPT_PATH_list[@]}"; do
    CKPT_PATH="${CKPT_PATH_list[$i]}"
    CONV_MODE="${conv_mode_list[$i]}"
    
    echo "Evaluating $CKPT_PATH with mode $CONV_MODE"
    
    CKPT_NAME=$(basename "$CKPT_PATH")
    
    # Get dataset list corresponding to current checkpoint
    eval "datasets=(${eval_dataset_list[$i]})"
    
    # Iterate through dataset list for current checkpoint
    for dataset in "${datasets[@]}"; do
        case $dataset in
            "mme")
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/eval/mme.sh "$CKPT_PATH" "$CKPT_NAME" "$CONV_MODE"
                ;;
            "gqa")
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/eval/gqa.sh "$CKPT_PATH" "$CKPT_NAME" "$CONV_MODE"
                ;;
            "textvqa")
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/eval/textvqa.sh "$CKPT_PATH" "$CKPT_NAME" "$CONV_MODE"
                ;;
            "pope")
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/eval/pope.sh "$CKPT_PATH" "$CKPT_NAME" "$CONV_MODE"
                ;;
            "mmvet")
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/eval/mmvet.sh "$CKPT_PATH" "$CKPT_NAME" "$CONV_MODE"
                ;;
            "sqa")
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/eval/sqa.sh "$CKPT_PATH" "$CKPT_NAME" "$CONV_MODE"
                ;;
            "mmbench")
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/eval/mmbench.sh "$CKPT_PATH" "$CKPT_NAME" "$CONV_MODE"
                ;;
        esac
    done
done
