#! /bin/bash
ivshmem_backend_path="/dev/shm/ivshmem-0-ub11.dat"

sudo groupadd $USER
sudo usermod -aG $USER $USER

# use filepath as-is, guaranteed by qemu + device config.
sudo chown root:$USER "$ivshmem_backend_path"
sudo chmod 666 $USER "$ivshmem_backend_path"
