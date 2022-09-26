#!/usr/bin/env bash
set -x

GPUS=${GPUS:-2}
PORT=${PORT:-29500}
if [ $GPUS -lt 2 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-2}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-2}

OUTPUT_DIR=$1
PRETRAINED_WEIGHTS=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT} --use_env \
main.py --with_box_refine --binary --freeze_text_encoder --visualize \
--epochs 6 --lr_drop 3 5 \
--output_dir=${OUTPUT_DIR} --pretrained_weights=${PRETRAINED_WEIGHTS} ${PY_ARGS}

# inference
#CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
#python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
#--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT}  ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"

