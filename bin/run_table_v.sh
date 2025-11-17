#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

BENCH_DIR=${CURRENT_DIR}/benchmarks/
RAW_DATA_DIR=${CURRENT_DIR}/raw_data/figure_7 # reuse the raw data from figure 7
RESULT_DIR=${CURRENT_DIR}/results/table_v
PY_DIR=${CURRENT_DIR}/python/table_v

mkdir -p ${RAW_DATA_DIR}
mkdir -p ${RESULT_DIR}

python3 ${PY_DIR}/process.py --log-folder ${RAW_DATA_DIR} &> ${RESULT_DIR}/table_v.log
