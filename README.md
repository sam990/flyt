# Flyt

Flyt is a elastic GPU provisioning framework for Virtual Machines. It is built on top of Cricket and uses the same virtualization layer. Flyt has a distributed framework that can be used to provision GPU resources over a cluster of GPUs.


For Flyt to be able to insert the virtualization layer, the CUDA application has to link dynamically to the CUDA APIs. For this, you have to pass `-cudart shared` to `nvcc` during linking.


# Dependencies
Flyt requires
- CUDA Toolkit (E.g. CUDA 12.1)
- `rpcbind`
- `libcrypto`
- `libtirpc`
- `patchelf`

libtirpc and patchelf built as part of the main Makefile.

On the system where the Cricket server should be executed, the appropriate NVIDIA drivers should be installed.

# Building

Before building FLyt, you need to edit the location of configuration files in `control-managers/src/common/config.rs`

```
git clone https://github.com/RWTH-ACS/cricket.git
cd cricket && git submodule update --init
LOG=INFO make
```

Environment variables for Makefile:
- `LOG`: Log level. Can be one of `DEBUG`, `INFO`, `WARNING`, `ERROR`.
- `WITH_DEBUG`: Use gcc debug flags for compilation

You can also build the binaries for the client and server separately by running `make install-cpu-client` or `make install-cpu-server`.

On the client you need to substiture the original cuda runtime library with the flyt runtime library. `make install-client-lib` can be used to install the client library. `make restore-client-lib` can be used to restore the original cuda runtime library.


# Running a CUDA Application
You need to run following modules:

1. **flyt-cluster-manager:** This can be run on any machine in the cluster. It is responsible for managing the cluster of GPUs.
2. **flyt-node-manager:** This should be run as a daemon on machines where the GPU is available. Before running the node manager, make sure that mps is enabled on the GPU. Also you need to set the environment variable ` ` before running the node manager.
3. **flyt-client-manager:** This should be run on the Virtual Machine as a daemon.

You should update the configuration files to point to the correct IP addresses/ports and other configurations.

Furthermore, a mongo database should be available to store the initial requirements of Virtual Machines. 
The configuration file for the cluster manager should be updated to point to the database.
A collection with the name `vm_required_resources` should be created in the database. The collection should have information in the following format:

```
{
    vm_ip: <The ip address of the Virtual Machine>,
    host_ip: <The ip address of the Host Machine of the VM>,
    compute_units: <The number of SM cores the VM should be allocated>,
    memory: <The amount of memory in GB the VM should be allocated>
}
```


Make sure that applications are linked with the shared cudart library. You can do this by passing `-cudart shared` to `nvcc` during linking.

You can launch the applications normally as any other CUDA application. The framework will take care of the rest.

# Provioning Control
`flytctl` is a command line tool to interact with the Flyt framework. It should be run on the cluster manager machine. 
Use `flytctl --help` to get more information about the commands.

# Contributing

## File structue
* **control-managers:** The control managers for the Flyt framework
* **cpu:** The virtualization layer
    * **libtirpc:** Transport Indepentend Remote Procedure Calls is requried for the virtualization layer
    * **patchelf:** A tool to modify the dynamic linker and RPATH of an executable
* **tests:** various CUDA applications to test cricket.


## Acknowledgments
This work was done in Synerg Lab at IIT Bombay with support from IBM Research, India.
