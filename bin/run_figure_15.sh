#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi


# get one argument from the command line
if [ $# -ne 1 ]; then
    echo "Error: Please provide the Megatron path"
    exit 1
fi

MEGATRON=$1

# check if pretrain_gpt.py exists under the Megatron path
if [ ! -f ${MEGATRON}/pretrain_gpt.py ]; then
    echo "Error: pretrain_gpt.py does not exist under the Megatron path"
    exit 1
fi

SCRIPT_PATH=${CURRENT_DIR}/bin/
BENCH_DIR=${CURRENT_DIR}/benchmarks/
RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_15
RESULT_DIR=${CURRENT_DIR}/results/figure_15
PY_DIR=${CURRENT_DIR}/python/figure_15

mkdir -p ${RAW_DATA_DIR}
mkdir -p ${RESULT_DIR}

cp -r ${SCRIPT_PATH}/run_dist_training.sh ${MEGATRON}

########################################################
# run data parallel training
########################################################

cd ${MEGATRON}
export MGPU_PROFILING=1
accelprof -v -t event_trace_mgpu ./run_dist_training.sh dp
mv run_dist_training.sh.accelprof.log ${RAW_DATA_DIR}/dp.accelprof.log

########################################################
# run tensor parallel training
########################################################

cd ${MEGATRON}
export MGPU_PROFILING=1
accelprof -v -t event_trace_mgpu ./run_dist_training.sh tp
mv run_dist_training.sh.accelprof.log ${RAW_DATA_DIR}/tp.accelprof.log

########################################################
# run pipeline parallel training
########################################################

cd ${MEGATRON}
export MGPU_PROFILING=1
accelprof -v -t event_trace_mgpu ./run_dist_training.sh pp
mv run_dist_training.sh.accelprof.log ${RAW_DATA_DIR}/pp.accelprof.log

########################################################
# process data
########################################################

mkdir -p ${RESULT_DIR}/dp
mkdir -p ${RESULT_DIR}/tp
mkdir -p ${RESULT_DIR}/pp

python3 ${PY_DIR}/process.py --log-file ${RAW_DATA_DIR}/dp.accelprof.log --output-folder ${RESULT_DIR}/dp
python3 ${PY_DIR}/process.py --log-file ${RAW_DATA_DIR}/tp.accelprof.log --output-folder ${RESULT_DIR}/tp
python3 ${PY_DIR}/process.py --log-file ${RAW_DATA_DIR}/pp.accelprof.log --output-folder ${RESULT_DIR}/pp


########################################################
# plot data
########################################################

python3 ${PY_DIR}/plot.py --log-path ${RESULT_DIR} --output-folder ${RESULT_DIR}
