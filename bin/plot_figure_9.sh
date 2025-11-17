#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

# get one argument from the command line
if [ $# -ne 1 ]; then
    echo "Error: Please provide the log folder"
    exit 1
fi

LOG_FOLDER=$1

RAW_DATA_DIR=${CURRENT_DIR}/${LOG_FOLDER}
PY_DIR=${CURRENT_DIR}/python/figure_9
RESULT_DIR=${CURRENT_DIR}/results/figure_9

mkdir -p ${RESULT_DIR}

python3 ${PY_DIR}/process.py --log-folder ${RAW_DATA_DIR} &> ${RESULT_DIR}/result.log

python3 ${PY_DIR}/plot.py --result-log ${RESULT_DIR}/result.log --output-folder ${RESULT_DIR}
