#!/bin/bash

# Log file path (adjust if needed)
LOG_FILE="readings.txt"

# Initialize an associative array to store first and last lines for each load
declare -A first_line last_line

# Read the log file line by line
while IFS= read -r line; do
    # Extract the Load (Threads) value using regex
    load=$(echo "$line" | grep -oP 'Load \(Threads\) : \K[0-9]+')

    # If the load value is found
    if [[ -n "$load" ]]; then
        # If this is the first time encountering this load, save it as the first line
        if [[ -z "${first_line[$load]}" ]]; then
            first_line[$load]="$line"
        fi
        # Always update the last line for this load
        last_line[$load]="$line"
    fi
done < "$LOG_FILE"

# Output the first and last lines for each load value
echo "First and Last lines for each unique Load (Threads):"
for load in "${!first_line[@]}"; do
    echo "Load (Threads): $load"
    echo "  First: ${first_line[$load]}"
    echo "  Last:  ${last_line[$load]}"
    echo
done

