#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

BENCH_DIR=${CURRENT_DIR}/benchmarks/
RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_13
RESULT_DIR=${CURRENT_DIR}/results/figure_13
PY_DIR=${CURRENT_DIR}/python/figure_13


########################################################
# collect data
########################################################

mkdir -p ${RAW_DATA_DIR}
mkdir -p ${RESULT_DIR}

MODEL_NAME="bert"

cd ${BENCH_DIR}/${MODEL_NAME}

profile_prefix="accelprof -v -t time_hotness_cpu"
run_model_cmd="python3 run_${MODEL_NAME}.py -t test"

$profile_prefix $run_model_cmd
mv run_${MODEL_NAME}.time_hotness_cpu.log ${RAW_DATA_DIR}/${MODEL_NAME}_time_hotness_cpu.log


########################################################
# process & plot
########################################################

python3 ${PY_DIR}/plot.py --result-log ${RAW_DATA_DIR}/${MODEL_NAME}_time_hotness_cpu.log --output-folder ${RESULT_DIR}

