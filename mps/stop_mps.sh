#!/bin/bash
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 2 -c DEFAULT
