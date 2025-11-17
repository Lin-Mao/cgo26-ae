#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

# get one argument from the command line
if [ $# -ne 1 ]; then
    echo "Error: Please provide the number of runs"
    exit 1
fi
NUM_RUNS=$1

RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_11
PY_DIR=${CURRENT_DIR}/python/figure_11
RESULT_DIR=${CURRENT_DIR}/results/figure_11
BENCH_DIR=${CURRENT_DIR}/benchmarks/

mkdir -p ${RAW_DATA_DIR}
mkdir -p ${RESULT_DIR}


#!/bin/bash

# profile all models
model_list=("alexnet" "resnet18" "resnet34" "bert" "gpt2" "whisper")

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=use_uvm:True

########################################################
# collect data
########################################################
LOG_FILE=${RAW_DATA_DIR}/uvm_advisor.log
echo "" > ${LOG_FILE}

UVM_ADVISOR_PATH=${CURRENT_DIR}/uvm-advisor

cd ${UVM_ADVISOR_PATH}
make -j

# no prefetch
echo "" >> ${LOG_FILE}
echo "-------------------------------- NO PREFETCH --------------------------------" >> ${LOG_FILE}
prefix_command="PREFETCH_MODE=0 LD_PRELOAD=${UVM_ADVISOR_PATH}/libop_callback_uvm.so" >> ${LOG_FILE}
for i in ${!model_list[@]}; do
    model=${model_list[$i]}
    cd ${BENCH_DIR}/$model
    run_command="python3 run_${model}.py -t test"
    echo "${prefix_command} ${run_command}" >> ${LOG_FILE}
    for ((j=1; j<=NUM_RUNS; j++)); do
        eval "${prefix_command} ${run_command}" >> ${LOG_FILE} 2>&1
    done
    cd ..
done

# object level prefetch
echo "" >> ${LOG_FILE}
echo "-------------------------------- OBJECT LEVEL PREFETCH --------------------------------" >> ${LOG_FILE}
prefix_command="PREFETCH_MODE=1 LD_PRELOAD=${UVM_ADVISOR_PATH}/libop_callback_uvm.so"
for i in ${!model_list[@]}; do
    model=${model_list[$i]}
    cd ${BENCH_DIR}/$model
    run_command="python3 run_${model}.py -t test"
    echo "${prefix_command} ${run_command}" >> ${LOG_FILE}
    for ((j=1; j<=NUM_RUNS; j++)); do
        eval "${prefix_command} ${run_command}" >> ${LOG_FILE} 2>&1
    done
    cd ..
done

# tensor level prefetch
echo "" >> ${LOG_FILE}
echo "-------------------------------- TENSOR LEVEL PREFETCH --------------------------------" >> ${LOG_FILE}
prefix_command="PREFETCH_MODE=2 LD_PRELOAD=${UVM_ADVISOR_PATH}/libop_callback_uvm.so"
for i in ${!model_list[@]}; do
    model=${model_list[$i]}
    cd ${BENCH_DIR}/$model
    run_command="python3 run_${model}.py -t test"
    echo "${prefix_command} ${run_command}" >> ${LOG_FILE}
    for ((j=1; j<=NUM_RUNS; j++)); do
        eval "${prefix_command} ${run_command}" >> ${LOG_FILE} 2>&1
    done
    cd ..
done


########################################################
# process data
########################################################

python3 ${PY_DIR}/process.py --log-folder ${RAW_DATA_DIR} &> ${RESULT_DIR}/result.log


########################################################
# plot figure
########################################################
python3 ${PY_DIR}/plot.py --result-log ${RESULT_DIR}/result.log --output-folder ${RESULT_DIR}