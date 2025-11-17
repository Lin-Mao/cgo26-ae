#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

BENCH_DIR=${CURRENT_DIR}/benchmarks/
RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_7

mkdir -p ${RAW_DATA_DIR}

model_list=(
    "alexnet"
    "resnet18"
    "resnet34"
    "bert"
    "gpt2"
    "whisper"
)

########################################################
# collect data
########################################################
cd ${BENCH_DIR}
profile_prefix="accelprof -v -t app_analysis"
for model in ${model_list[@]}; do
    cd ${BENCH_DIR}/${model}

    run_model_cmd="python3 run_${model}.py"

    # training
    $profile_prefix $run_model_cmd -t train
    mv run_${model}_app_analysis.log ${RAW_DATA_DIR}/train_${model}_app_analysis.log

    # inference
    $profile_prefix $run_model_cmd -t test
    mv run_${model}_app_analysis.log ${RAW_DATA_DIR}/test_${model}_app_analysis.log
done

########################################################
# process data
########################################################
RESULT_DIR=${CURRENT_DIR}/results/figure_7
PY_DIR=${CURRENT_DIR}/python/figure_7

python3 ${PY_DIR}/process.py --log-folder ${RAW_DATA_DIR} --output-folder ${RESULT_DIR}

########################################################
# plot figure
########################################################
python3 ${PY_DIR}/plot.py --log-folder ${RESULT_DIR} --output-folder ${RESULT_DIR}
