#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

BENCH_DIR=${CURRENT_DIR}/benchmarks/
RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_10
RESULT_DIR=${CURRENT_DIR}/results/figure_9
PY_DIR=${CURRENT_DIR}/python/figure_9

# check if the raw data directory exists
if [ ! -d ${RAW_DATA_DIR} ]; then
    echo "Running run_figure_10.sh to collect data first..."
    exit 1
fi


mkdir -p ${RESULT_DIR}


########################################################
# collect data
########################################################

# Use run_figure_10.sh to collect data to save time

########################################################
# process data
########################################################
python3 ${PY_DIR}/process_high_sample_rate.py --log-folder ${RAW_DATA_DIR} &> ${RESULT_DIR}/raw_result.log


########################################################
# plot figure
########################################################

python3 ${PY_DIR}/plot_single.py --result-log ${RESULT_DIR}/raw_result.log --output-folder ${RESULT_DIR}
