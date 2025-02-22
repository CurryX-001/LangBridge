#!/bin/bash

CKPT_PATH_list=(
    # "/scratch/Codebook/Codebook_weight/checkpoints/llama3_8b_vocab-19200-finetune"
    # "/scratch/Codebook/Codebook_weight/checkpoints/qwen2_5_7b_finetune_baseline_pretrain_lr_1e-3"
    # "/scratch/Codebook/Codebook_weight/checkpoints/qwen2_5_7b_finetune_vocab_19200_use_qwen2_0.5b_pretrain"
    # "/scratch/Codebook/Codebook_weight/checkpoints/qwen2_5_7b_finetune_vocab_19200_use_self_pretrain"
    # "/scratch/Codebook/Codebook_weight/checkpoints/qwen2_7b_finetune_baseline_pretrain_lr_1e-3"
    # "/scratch/Codebook/Codebook_weight/checkpoints/qwen2_7b_finetune_vocab_19200_llama3_8b_pretrain"
    # "/scratch/Codebook/Codebook_weight/checkpoints/qwen2_7b_finetune_vocab_25600_qwen2_sft_pretrain"
    # "/scratch/Codebook/Codebook_weight/checkpoints/qwen2_7b_finetune_vocab_32000_qwen2_sft_pretrain"
    "/scratch/Codebook/ACL/qwen2_7b_sft_vocab_25600_hidden_size_5120_use_self_pretrain"
)

conv_mode_list=(
    # "llama3"
    # "qwen_2"
    # "qwen_2"
    # "qwen_2"
    # "qwen_2"
    # "qwen_2"
    # "qwen_2"
    # "qwen_2"
    "qwen_2"
)

eval_dataset_list=(
    # "mmvet"
    # "mmvet mmbench"
    # "mmvet mmbench"
    # "mme gqa textvqa pope mmvet sqa mmbench"
    # "mmvet mmbench"
    # "mmvet"
    # "mme gqa textvqa pope mmvet sqa mmbench"
    "mme gqa textvqa pope mmvet sqa mmbench"
)

for i in "${!CKPT_PATH_list[@]}"; do
    CKPT_PATH="${CKPT_PATH_list[$i]}"
    CONV_MODE="${conv_mode_list[$i]}"
    
    echo "Evaluating $CKPT_PATH with mode $CONV_MODE"
    
    CKPT_NAME=$(basename "$CKPT_PATH")
    
    # 获取当前检查点对应的数据集列表
    eval "datasets=(${eval_dataset_list[$i]})"
    
    # 遍历当前检查点的数据集列表
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
