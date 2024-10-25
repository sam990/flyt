#!/bin/bash

# Read the log file and extract the desired lines
awk '/Execution/ {print prev_time, $0} {prev_time=$1}' log.txt

