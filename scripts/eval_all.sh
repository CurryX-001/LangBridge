#!/bin/bash

CKPT_PATH_list=(
"/mnt/data/jiaqi.liao/Codebook/checkpoints/llama3_it_8b_it_vocab_12800_dim_1024_finetune_mmvocab_with_qwen_0.5b_it_vocab_12800_dim_1024_finetuned_mlp"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/llama3_it_8b_it_vocab_12800_dim_1024_finetune_mmvocab_with_qwen_0.5b_it_vocab_12800_dim_1024_pretrained_mlp"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2_it_8b_it_vocab_12800_dim_1024_finetune_mmvocab_with_llama3_8b_it_vocab_12800_dim_1024_finetune"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2_it_8b_it_vocab_12800_dim_1024_finetune_mmvocab_with_llama3_8b_it_vocab_12800_dim_1024_pretrain"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2_it_8b_it_vocab_12800_dim_1024_finetune_mmvocab_with_qwen_0.5b_it_vocab_12800_dim_1024_finetuned_mlp"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2_it_8b_it_vocab_12800_dim_1024_finetune_mmvocab_with_qwen_0.5b_it_vocab_12800_dim_1024_pretrain"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2.5_it_7b_it_vocab_12800_dim_1024_finetune_mmvocab_with_llama3_8b_it_vocab_12800_dim_1024_finetune"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2.5_it_7b_it_vocab_12800_dim_1024_finetune_mmvocab_with_llama3_8b_it_vocab_12800_dim_1024_pretrain"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2.5_it_7b_it_vocab_12800_dim_1024_finetune_mmvocab_with_qwen_0.5b_it_vocab_12800_dim_1024_finetuned_mlp"
"/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2.5_it_7b_it_vocab_12800_dim_1024_finetune_mmvocab_with_qwen_0.5b_it_vocab_12800_dim_1024_pretrained_mlp"
)

conv_mode_list=(
    "llama3"
    "llama3"
    "qwen_2"
    "qwen_2"
    "qwen_2"
    "qwen_2"
    "qwen_2"
    "qwen_2"
    "qwen_2"
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
