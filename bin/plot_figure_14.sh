#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

BENCH_DIR=${CURRENT_DIR}/benchmarks/
RESULT_DIR=${CURRENT_DIR}/results/figure_14
PY_DIR=${CURRENT_DIR}/python/figure_14


mkdir -p ${RESULT_DIR}


# plot the memory usage comparison
python ${PY_DIR}/plot_cmp.py --log-path ${CURRENT_DIR}/pre-results/amd-nvidia-compare --output-folder ${RESULT_DIR}