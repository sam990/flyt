#!/bin/bash

mem_size=256

counter=14
csv_all='size,r1,r2,r3,r4'


while [ $counter -ne 0 ]
do
echo "Mem size: $mem_size"
row="$mem_size"
for i in {0..3}; do
	echo "$i"
	time=`./benchE $mem_size 8192 5000 | tee -a benchE.log |  sed -n  's/Time: \([[:digit:]]\+\) ms/\1/p'`
	row="$row,$time"
done
echo "$row"
csv_all="$csv_all\n$row"
mem_size=$(( mem_size * 2 ))
counter=$(( counter - 1 ))
done
echo -e  "$csv_all"