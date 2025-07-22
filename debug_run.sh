#!/bin/bash

set -euo pipefail
CONFIG_FILE="/home/jeromeku/torchtitan/torchtitan/experiments/qwen3_moe/train_configs/debug.toml"
NGPU=1
MODULE="torchtitan.debug_train"
DEBUG_DISTRIBUTED=1

if [[ ${DEBUG_DISTRIBUTED} == 0 ]]; then
    CMD="torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" -m ${MODULE} --job.config_file ${CONFIG_FILE}"
else
    CMD="LOCAL_RANK=0 python -m ${MODULE} --job.config_file ${CONFIG_FILE}"
fi

echo "${CMD}"
eval "${CMD}"