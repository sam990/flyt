#!/bin/bash
# Copyright (c) 2024-2026 SynerG Lab, IITB

# the following must be performed with root privilege
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
