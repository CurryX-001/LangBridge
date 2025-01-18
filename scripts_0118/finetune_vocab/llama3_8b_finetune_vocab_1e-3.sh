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

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=37229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch


torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  /mnt/petrelfs/mengfanqing/Codebook/llava/train/train_mem.py \
    --deepspeed /mnt/petrelfs/mengfanqing/Codebook/scripts/zero3.json \
    --model_name_or_path /mnt/hwfile/gveval/mfq/Meta-Llama-3-8B-Instruct \
    --version llama3 \
    --data_path /mnt/hwfile/gveval/mfq/playground/filtered_llava_v1_5_mix665k.json\
    --image_folder /mnt/hwfile/mllm/chenlin/llava/data\
    --vision_tower /mnt/hwfile/gveval/mfq/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /mnt/petrelfs/mengfanqing/Codebook/checkpoints/Llama-3-8B-pretrain-3-vocab19200-1e-3/mm_projector.bin\
    --mm_projector_type mm_vocab \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /mnt/hwfile/gveval/mfq/checkpoints/Llama-3-8B-finetune-vocab19200-1e-3\
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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
    --mm_vocab_size 19200 \
    --mlp_hidden_size 5120 \
    --mm_vocab_matrix "/mnt/petrelfs/mengfanqing/Codebook/embedding_matrix_llm/Meta-Llama-3-8B-Instruct_embedding_matrix_19200.pt"
            2>&1 | tee $LOG_FILE
