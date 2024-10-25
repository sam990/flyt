#!/bin/bash

# Usage: <script> <result-filename-base> <number of processes to run in parallel> <sync-rate = 0 for every call, 100 for no call till end>
# The file <result-filename-base>_$i.csv will store the kernel completion latencies
# for the ith ptocess.


num_processes=$2
shm_name="/benchR_shm"

mkfifo pipe-read
mkfifo pipe-write

tail -f pipe-write | ./synch $shm_name $num_processes >pipe-read  &

read -N1 ch <pipe-read

echo "$ch: SHM and barrier initialised"


mem_size=25
iters=5000

filename_base=$1
sync_rate=$3

echo "$filename_base" 
echo "processes"
echo "$num_processes"

minus_num_processes=$((num_processes - 1))
process=()
mystart=$(date +%s%N)

for ((i=1;i<=$minus_num_processes;i++)) do
   ./benchR $mem_size 512 $iters $sync_rate "$filename_base"_"$i.csv" $shm_name &
   pid=$!
   echo "iteration is $i"
   process+=($pid)
   #CRICKET_RPCID=1 REMOTE_GPU_ADDRESS=10.129.27.234 LD_PRELOAD=/home/vm/cricket/bin/cricket-client.so ./benchR $mem_size 512 $iters "$filename_base"_"$i.csv" $shm_name &
done

# Check if the correct number of arguments is provided
if [ "$sync_rate" -ne 0 ]; then
   ./benchR $mem_size 512 $iters 0 "$filename_base"_"$i.csv" $shm_name &
   pid=$!
   process+=($pid)
fi
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





