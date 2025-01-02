#!/bin/bash
set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=38659
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME=bond0

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  /mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/llava/train/train_mem.py \
    --deepspeed /mnt/petrelfs/mengfanqing/LLaVA_ori_1104/LLaVA-main/scripts/zero2.json \
    --model_name_or_path /mnt/petrelfs/share_data/mfq/Qwen2-0.5B-Instruct \
    --version plain \
    --data_path /mnt/petrelfs/share_data/wangwenhai/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/petrelfs/share_data/wangwenhai/playground/data/LLaVA-Pretrain/images \
    --vision_tower /mnt/petrelfs/share_data/wangwenhai/llm/clip-vit-large-patch14-336 \
    --mm_projector_type mm_vocab \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/Qwen2-0.5B-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --mm_vocab_size 19200 \
    --mlp_hidden_size 5120 \
    --mm_vocab_matrix "/mnt/petrelfs/mengfanqing/MM_vocab/embedding_matrix_vocab_19200/Qwen2-0.5B-Instruct_embedding_matrix_vocab_19200.pt"
