#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi


BENCH_DIR=${CURRENT_DIR}/benchmarks/
RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_14
RESULT_DIR=${CURRENT_DIR}/results/figure_14
PY_DIR=${CURRENT_DIR}/python/figure_14

mkdir -p ${RAW_DATA_DIR}
mkdir -p ${RESULT_DIR}

MODEL_NAME="gpt2"

cd ${BENCH_DIR}/${MODEL_NAME}

profile_prefix="accelprof -v -t app_analysis"
run_model_cmd="python3 run_${MODEL_NAME}.py"

$profile_prefix $run_model_cmd
mv run_${MODEL_NAME}.accelprof.log ${RAW_DATA_DIR}/${MODEL_NAME}_app_analysis_rocm.log

python3 ${PY_DIR}/process_nvidia.py --log-file ${RAW_DATA_DIR}/${MODEL_NAME}_app_analysis_rocm.log &> ${RESULT_DIR}/out_nvidia.log
