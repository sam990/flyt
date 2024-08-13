#!/bin/bash

ip link add veth0 link eno0 type macvlan mode bridge
ip addr add dev veth0 10.129.27.234/16
ip link set veth0 up
