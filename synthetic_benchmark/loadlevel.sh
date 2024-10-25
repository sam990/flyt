#!/bin/bash

# Read the log file and extract the desired lines
awk '
{
    # Extract the "Load (Threads)" value
    match($0, /Load \(Threads\) : [0-9]+/, arr)
    threads = arr[0]

    # Print the first occurrence of each unique "Load (Threads)" value
    if (threads != last_threads) {
        print $0
        last_threads = threads
    }
}' readings.txt

