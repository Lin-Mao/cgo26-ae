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

pip install gdown

gdown https://drive.google.com/uc?id=1ETHKAvit9BurcypF8ZGm6fj4ORCdpxLg
unzip pre-results.zip && rm pre-results.zip


# Create directories
mkdir raw_data
mkdir results
