#!/bin/bash

# List of ivshmem backend paths
ivshmem_backend_paths=(
    "/dev/shm/ivshmem-0-ub11.dat"
    "/dev/shm/ivshmem-1-ub11.dat"
    #"/dev/shm/ivshmem-2-ub11.dat"
    # Add more paths as needed
)

# Create a group for the current user if it doesn't exist
sudo groupadd $USER 2>/dev/null
sudo usermod -aG $USER $USER

# Iterate over each backend path in the list
for path in "${ivshmem_backend_paths[@]}"; do
    # Change the ownership and permissions of the backend path
    if [ -e "$path" ]; then
        echo "Processing $path"
        sudo chown root:$USER "$path"
        sudo chmod 666 "$path"
    else
        echo "Warning: $path does not exist"
    fi
done
