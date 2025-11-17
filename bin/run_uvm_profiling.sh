#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

model_list=("alexnet" "resnet18" "resnet34" "bert" "gpt2" "whisper")

BENCH_DIR=${CURRENT_DIR}/benchmarks/

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=use_uvm:True
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

profile_prefix="accelprof -v -t uvm_advisor"
for model in ${model_list[@]}; do
    cd ${BENCH_DIR}/${model}

    run_model_cmd="python3 run_${model}.py -t test"

    # inference
    $profile_prefix $run_model_cmd
done
