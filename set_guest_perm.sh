#! /bin/bash

sudo chown root:$USER /sys/
sudo chmod 666 /sys/bus/pci/devices/0000:08:01.0/resource2 # needs to be different ,ie. = pci_path