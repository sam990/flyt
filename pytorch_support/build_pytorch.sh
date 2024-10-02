#!/bin/bash

# This script is used to configure the build environment for PyTorch 
# to work with Flyt

if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-pytorch>"
    exit 1
fi

PYTORCH_DIR=$1

if [ ! -d "$PYTORCH_DIR" ]; then
    echo "Error: $PYTORCH_DIR does not exist"
    exit 1
fi

SCRIPT_DIR=`pwd`

# set the environment variables
source $CURRENT_DIR/build_env.sh

# Go to the PyTorch directory
cd $PYTORCH_DIR
python setup.py build --cmake-only

# configure and generate the build files
cmake -DCUDA_NVCC_FLAGS="-cudart=shared" build

# install pytorch
python setup.py install
