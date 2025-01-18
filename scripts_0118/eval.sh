#!/bin/bash
set -x

GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTA_TYPE=${4}
PARTITION=${5}
SCRIPT=${6}
MODEL_PATH=$7    # /mnt/petrelfs/mengfanqing/Codebook/checkpoints/Qwen2-7B-finetune_vocab_qwen0.5_pre
CONV=${8}        # qwen_2
SCRIPTNAME=$(basename ${SCRIPT} .sh)
SCRIPTSUBDIR=$(basename $(dirname ${SCRIPT}))

# PARTITION=${PARTITION:-"VC3"}
# GPUS=${GPUS:-8}
# GPUS_PER_NODE=${GPUS_PER_NODE:-8}
# QUOTA_TYPE=${QUOTA_TYPE:-"spot"}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

SRUN_ARGS=${SRUN_ARGS:-" --jobid=3851832"} # 3713373 3700256

# BATCH_SIZE=${BATCH_SIZE:-256}
# PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
# GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export MASTER_PORT=32423
export TF_CPP_MIN_LOG_LEVEL=3
# unset CUDA_LAUNCH_BLOCKING
# export CUDA_LAUNCH_BLOCKING=1

WORK_DIR="$(pwd)/output/${SCRIPTSUBDIR}"
# WORK_DIR='/mnt/petrelfs/tianchangyao.p/code/lmm-integration/embodied_foundation-2538f3/InternVL/internvl_chat'
OUTPUT_DIR="${WORK_DIR}/${SCRIPTNAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


SUFFIX=$(date '+%Y%m%d%H%M')

srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  bash ${SCRIPT}  ${MODEL_PATH} ${CONV}\
  2>&1 | tee -a "${OUTPUT_DIR}/training_${SUFFIX}.log"
