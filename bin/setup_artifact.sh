#!/bin/bash

CURRENT_DIR=$(pwd)
if [ $(basename $CURRENT_DIR) != "cgo26-ae" ]; then
    echo "Error: Please run this script in the cgo26-ae directory"
    exit 1
fi

git clone https://github.com/AccelProf/benchmarks.git


# Install dependencies
pip install regex
pip install matplotlib
pip install circlify
pip install pandas
pip install git+https://github.com/openai/whisper.git


# Create directories
mkdir raw_data
mkdir results
