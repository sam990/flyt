#!/bin/bash

# Usage: <script> <result-filename-base> <number of processes to run in parallel>
# The file <result-filename-base>_$i.csv will store the kernel completion latencies
# for the ith ptocess.


num_processes=$2
shm_name="/benchF_shm"

mkfifo pipe-read
mkfifo pipe-write

tail -f pipe-write | ./synch $shm_name $num_processes >pipe-read  &

read -N1 ch <pipe-read

echo "$ch: SHM and barrier initialised"


mem_size=25
iters=5000

filename_base=$1

echo "$filename_base" 
echo "processes"
echo "$num_processes"

process=()
mystart=$(date +%s%N)

for ((i=1;i<=$num_processes;i++)) do
   ./benchF $mem_size 512 $iters "$filename_base"_"$i.csv" $shm_name &
   pid=$!
   process+=($pid)
   #CRICKET_RPCID=1 REMOTE_GPU_ADDRESS=10.129.27.234 LD_PRELOAD=/home/vm/cricket/bin/cricket-client.so ./benchF $mem_size 512 $iters "$filename_base"_"$i.csv" $shm_name &
done

echo "1" >pipe-write

rm pipe-read
rm pipe-write

# Wait for each child process to complete
for pid in "${process[@]}"; do
    echo "waiting for pid $pid"
    wait $pid
done

myend=$(date +%s%N)
echo "Rate time was `expr $myend - $mystart` nanoseconds."  > "$filename_base"_"num_processes"_512_5000.txt





