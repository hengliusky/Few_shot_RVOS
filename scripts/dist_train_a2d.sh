#!/usr/bin/env bash
set -x

GPUS=${GPUS:-3}
PORT=${PORT:-29500}
if [ $GPUS -lt 3 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-3}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=$1
PRETRAINED_WEIGHTS=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this
echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# train & test
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 1 \
--epochs 6 --lr_drop 3 5 \
--output_dir=${OUTPUT_DIR} --pretrained_weights=${PRETRAINED_WEIGHTS} ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"
echo ${PY_ARGS}

