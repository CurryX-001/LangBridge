#!/bin/bash
set -x
# 获取当前运行的 .sh 文件名（不包括路径）
SCRIPT_NAME=$(basename "$0")
SCRIPT_NAME_NO_EXT="${SCRIPT_NAME%.*}"

# 获取当前时间并格式化
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./output/${SCRIPT_NAME_NO_EXT}"
LOG_FILE="${LOG_DIR}/out_${TIMESTAMP}.log"

# 创建日志目录（如果它不存在）
mkdir -p $LOG_DIR

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=37226
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_SOCKET_IFNAME=bond0

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  /mnt/petrelfs/mengfanqing/Codebook/llava/train/train_mem.py \
    --deepspeed /mnt/petrelfs/mengfanqing/Codebook/scripts/zero2.json \
    --model_name_or_path /mnt/hwfile/gveval/mfq/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path /mnt/hwfile/gveval/mengfanqing/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/hwfile/gveval/mengfanqing/LLaVA-Pretrain/images \
    --vision_tower /mnt/hwfile/gveval/mfq/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/Llama-3-8B-pretrain-baseline-1e-3_1014 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate  1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
        2>&1 | tee $LOG_FILE
