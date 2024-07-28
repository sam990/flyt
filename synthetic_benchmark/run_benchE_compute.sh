#!/bin/bash

mem_size=64

counter=9
csv_all='iters,r1,r2,r3,r4'
iters=16000

while [ $counter -ne 0 ]
do
echo "iters: $iters"
row="$iters"
for i in {0..3}; do
	# echo $i
	time=`./benchE $mem_size 8192 $iters | tee -a benchE.log |  sed -n  's/Time: \([[:digit:]]\+\) ms/\1/p'`
	row="$row,$time"
done
echo "$row"
csv_all="$csv_all\n$row"
iters=$(( iters + 1000 ))
counter=$(( counter - 1 ))
done
echo -e  "$csv_all"
