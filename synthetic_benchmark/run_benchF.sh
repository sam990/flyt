#!/bin/bash

num_processes=$2
shm_name="/benchF_shm"

mkfifo pipe-read
mkfifo pipe-write

tail -f pipe-write | ./synch $shm_name $num_processes >pipe-read  &

read -N1 ch <pipe-read

echo "$ch: SHM and barrier initialised"


mem_size=256
iters=5000

filename_base=$1

for ((i=1;i<=$num_processes;i++)) do
    ./benchF $mem_size 512 $iters "$filename_base"_"$i.csv" $shm_name &
done

echo "1" >pipe-write

rm pipe-read
rm pipe-write




