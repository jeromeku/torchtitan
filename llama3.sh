REPO_ROOT=$(pwd)
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-1}
STEPS=${STEPS:-100}
NGPU=${NGPU:-1}
MX_RECIPE=${MX_RECIPE:-"mxfp8_cublas"}
MODEL_FLAVOR=${MODEL_FLAVOR:-"debug_model"}
# temporary log file which is deleted after performance data is parsed out and metrics are calculated.
DATETIME=$(date +%Y%m%d_%H%M)
LOG_FILE="${REPO_ROOT}/logs/${DATETIME}.${MX_RECIPE}.log"

# validate recipe name
if [ -n "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" ] && [ -n "${MX_RECIPE}" ]; then
    echo "Error: both FLOAT8_RECIPE_WITH_BEST_SETTINGS and MX_RECIPE are set, please only set one of them." >&2
    exit 1
elif [ -n "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" ]; then
  if [ "${FLOAT8_RECIPE_WITH_BEST_SETTINGS}" == "tensorwise" ]; then
    FLOAT8_ARGS="--model.converters="float8" --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp"
  else
    FLOAT8_ARGS="--model.converters="float8" --float8.recipe_name=${FLOAT8_RECIPE_WITH_BEST_SETTINGS}"
  fi
elif [ -n "${MX_RECIPE}" ]; then
    FLOAT8_ARGS="--model.converters="quantize.linear.mx" --quantize.linear.mx.recipe_name=${MX_RECIPE}"
else
    FLOAT8_ARGS=""
fi

echo "float8 args: ${FLOAT8_ARGS}"

# run the command with the specified arguments
CONFIG_FILE="${REPO_ROOT}/torchtitan/models/llama3/train_configs/${MODEL_FLAVOR}.toml" 
RUN_CMD="CONFIG_FILE=${CONFIG_FILE} NGPU=${NGPU} ./run_train.sh --training.steps=${STEPS} --training.local-batch-size=${LOCAL_BATCH_SIZE} --compile.enable ${FLOAT8_ARGS} ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}"

echo ${RUN_CMD}
eval ${RUN_CMD} 2>&1 | tee ${LOG_FILE}

# # return to original working directory
# cd $original_dir

# # parse logs to calculate top line metrics
# python benchmarks/float8/training/parse_torchtitan_logs.py --log-file ${LOG_FILE}

# # # clean up logs
# # rm ${LOG_FILE}
