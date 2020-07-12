#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

export CUDA_VISIBLE_DEVICES=0,1; python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

