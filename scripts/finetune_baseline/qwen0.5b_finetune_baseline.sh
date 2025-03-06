#!/bin/bash
set -x

# 添加时间戳变量
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 创建日志目录
LOG_DIR="./checkpoints/qwen2_0.5b_it_vocab_12800_dim_1024_pretrain_2e-5_finetune"
mkdir -p ${LOG_DIR}
# 设置日志文件路径
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# 将命令的输出重定向到日志文件
{
    echo "开始训练时间: $(date)"
    echo "GPUS: ${GPUS}"
    echo "BATCH_SIZE: ${BATCH_SIZE}"
    echo "PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
    echo "GRADIENT_ACC: ${GRADIENT_ACC}"
    
torchrun --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=6105 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --version qwen_2 \
    --data_path playground/data/llava_v1_5_mix665k_filtered.json \
    --image_folder playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2-0.5b_it_vocab_12800_dim_1024_pretrain_2e-5/mm_projector.bin \
    --mm_projector_type mm_vocab \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${LOG_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --mm_vocab_size 12800 \
    --mlp_hidden_size 1024 \
    --mm_vocab_matrix "/mnt/data/jiaqi.liao/Codebook/Qwen2-0.5B-Instruct_embedding_matrix_12800.pt"

    echo "结束训练时间: $(date)"
} > "${LOG_FILE}" 2>&1 &