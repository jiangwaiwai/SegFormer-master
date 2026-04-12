#!/usr/bin/env bash

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 CONFIG GPUS [PY_ARGS...]"
    exit 1
fi

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
