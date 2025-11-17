#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename ${CURRENT_DIR}) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

RAW_DATA_DIR=${CURRENT_DIR}/raw_data
RESULT_DIR=${CURRENT_DIR}/results

# clean Figure 7
echo "Cleaning Figure 7..."
rm -rf ${RAW_DATA_DIR}/figure_7
rm -rf ${RESULT_DIR}/figure_7


# clean Table V
echo "Cleaning Table V..."
rm -rf ${RAW_DATA_DIR}/table_v
rm -rf ${RESULT_DIR}/table_v