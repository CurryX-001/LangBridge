set -x
# export TRANSFORMERS_VERBOSITY=info


GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTATYPE=${4}
PARTITION=${5}

# GPUS=${GPUS:-8}
# GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODE=1
else
    NODE=$[GPUS/GPUS_PER_NODE]
fi

SCRIPT=${6}
CKPT_PATH=${7}

CKPT=`basename ${CKPT_PATH}`
SCRIPTNAME=`basename ${SCRIPT} .sh`
DIR=./checkpoints/eval/${CKPT}/${SCRIPTNAME}
mkdir -p ${DIR}
SUFFIX=`date '+%Y%m%d%H%M'`

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

SRUN_ARGS=${SRUN_ARGS:-""}

PY_ARGS=${PY_ARGS:-""}

# export MASTER_PORT=22116
srun -p ${PARTITION} \
    --quotatype=${QUOTATYPE} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    bash ${SCRIPT} ${CKPT_PATH} \
    ${@:8} ${PY_ARGS} 2>&1 
    #| tee -a ${DIR}/${SCRIPTNAME}_${SUFFIX}.log
