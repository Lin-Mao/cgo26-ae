#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

BENCH_DIR=${CURRENT_DIR}/benchmarks/
RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_9

mkdir -p ${RAW_DATA_DIR}

model_list=(
    "alexnet"
    "resnet18"
    "resnet34"
    "bert"
    "gpt2"
    "whisper"
)

# run the application without any tool
prefix_command="accelprof -v -t none"
for i in ${!model_list[@]}; do
    model=${model_list[$i]}
    cd ${BENCH_DIR}/$model
    echo "[Execution] model: $model"
    run_model_cmd="python3 run_${model}.py -t test"
    ${prefix_command} ${run_model_cmd}
    mv run_${model}.accelprof.log ${RAW_DATA_DIR}/test_run_${model}.accelprof.log
done

# run the application with app_analysis with GPU analysis
prefix_command="accelprof -v -t app_analysis"
for i in ${!model_list[@]}; do
    model=${model_list[$i]}
    cd ${BENCH_DIR}/$model
    echo "[GPU] model: $model"
    run_model_cmd="python3 run_${model}.py -t test"
    ${prefix_command} ${run_model_cmd}
    mv run_${model}.accelprof.log ${RAW_DATA_DIR}/test_gpu_${model}.accelprof.log
done

# sample_rate_list=("30" "30" "50" "10" "10" "10")
sample_rate_list=("150" "150" "250" "50" "50" "50")

# run the application with app_analysis_cpu
prefix_command="accelprof -v -t app_analysis_cpu"
for i in ${!model_list[@]}; do
    model=${model_list[$i]}
    rate=${sample_rate_list[$i]}
    cd ${BENCH_DIR}/$model
    echo "[CPU] model: $model, sample_rate: $rate"
    export ACCEL_PROF_ENV_SAMPLE_RATE=$rate
    run_model_cmd="python3 run_${model}.py -t test"
    ${prefix_command} ${run_model_cmd}
    mv run_${model}.accelprof.log ${RAW_DATA_DIR}/test_cpu_${model}.accelprof.log
done

# run the application with app_analysis_nvbit
prefix_command="accelprof -v -d nvbit -t app_analysis"
for i in ${!model_list[@]}; do
    model=${model_list[$i]}
    rate=${sample_rate_list[$i]}
    cd ${BENCH_DIR}/$model
    echo "[NVBIT] model: $model, sample_rate: $rate"
    export ACCEL_PROF_ENV_SAMPLE_RATE=$rate
    run_model_cmd="python3 run_${model}.py -t test"
    ${prefix_command} ${run_model_cmd}
    mv run_${model}.accelprof.log ${RAW_DATA_DIR}/test_nvbit_${model}.accelprof.log
done


########################################################
# process data
########################################################




########################################################
# plot figure
########################################################

