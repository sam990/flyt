#!/bin/bash
# Copyright (c) 2024-2026 SynerG Lab, IITB

echo quit | nvidia-cuda-mps-control
nvidia-smi -i 2 -c DEFAULT
