Usage Modes
===========
RPC-only: Several VMs running on a physical machine with no GPU. CUDA processes send RPC commands to a cluster of GPUs. A GPU cluster has a single public endpoint (cluster manager IP) which receives messages from VMs. [WORKING]

Shared-mem: Several VMs running on a physical Machine with a GPU. CUDA processes running on VM sends commands via an RPC channel to an end-point on
the host, MPS-enabled machine. Data-intensive CUDA commands (`cudaMemcpy`) transfer data via a shared-memory channel. [IN-PROGRESS]

Build-And-Test Prerequisites:
-----------------------------
Flyt root directory: `flyt/`. All filepaths are relative to the root directory.
- Three kinds of nodes: Client node (VM running cuda app), Cluster Node (A machine with mps GPU), cluster-manager node (public endpoint, communicates with every cluster node). 
- The "cluster manager" node may be one of the cluster nodes or a regular machine.
- Each cluster node must support nvidia MPS with environment variable `CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1`

Two possible setups:

A. Independently build Flyt from source code on each client node, each cluster-node and the cluster-manager node.

B. Build Flyt from source via `make` on the cluster-manager node and share the `/bin`, `/configs` to the other nodes. Modify the `LD_LIBRARY_PATH` variable on client and cluster nodes to search the `/bin` folder for custom dynamically loaded libraries.

## Build-and-Test Setup A:
Independently build Flyt from source code on each client node, each cluster-node and the cluster-manager node.

(I) Do Cluster-Manager Node
---------------------------
Single network endpoint for the GPU cluster node(s).
On local machine
1. Build Flyt control managers: `make install-cmgrs`
- `bin/flytctl` and `bin/flyt-cluster-manager` eventually run on the cluster-manager node.

2. Initialise MongoDB database with network endpoint details of the client VM.
- Install mongodb
- `npm link mongodb`
- `sudo apt install mongodb`
- `sudo systemctl start mongod`
Create the database
-------------------
mongosh
use flyt
db.createUser({
    user: "adminUser",
    pwd: "flyt",
    roles: [{ role: "readWrite", db: "flyt" }]
})
- exit
- sudo rm -rf /tmp/mongodb-27017.sock (if ECONNREFUSED)
- `node setup/node_mongo_insert.js`.

3. Ensure `/configs/cluster-mgr-config.toml` is set to the appropriate values.
- [vm-resource-db]: MongoDB login credentials and network endpoints, used by cluster manager to connect to the databade and access VM info.
- [ports]: One listen port for each client and cluster node.
- [virt-server-auto-deallocate]: Remove a cluster-node from available pool after a timeout
- [ipc]: Message queue for mongoDB <-> cluster manager comms and a socket for flyctl->cluster manager comms.

4. Start the flyt-cluster-manager: `./bin/flyt-cluster-manager`
- Logs into mongoDB database server with configured credentials.
- Obtains VM data from the mongodb server via a message queue `/tmp/flyt-rmgr-queue`
- Listens to Client Node(s) and Cluster Node(s) at configured ports.

(II) Do Cluster Node
--------------------
Runs the flyt backend framework.
SHM Mode: ub-12-3 host, RPC Mode: ub-11 host
1. Build the multithreaded RPC server: `make install-cpu-server`
- Subsequently decodes incoming RPCs from client(s), calls the regular cuda runtime-api functions, and transmits results back to the client.

2. Build the flyt control managers: `make install-cmgrs`
- `flyt-node-manager` daemon eventually runs on each cluster (GPU) node.

3. Ensure `configs/servnode-config.toml` is set to appropriate values
- [resource-manager]: address of cluster manager + its cluster node listen port.
- [virt-server]: cricket-rpc-server path, thread mode.
- [ipc]: message queue backend for flyt-node-manager <-> cricket-rpc-server comms

4. Start MPS daemon `bash mps/start_mps.sh`
- Ensure environment variable `CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1` is set.

5. Start the flyt-node-manager `./bin/flyt-node-manager`
- Starts the rpc-server, listens for new clients.


(III) Do Client Node
--------------------
Runs CUDA applications linked to virtualised cuda runtime library.
- ub-12-3 VM.
1. Build virtualised CUDA runtime library and replace the standard cuda runtime API library in kernel: `make install-client-lib`.
- Copies virtualised library to default cuda library location in VM kernel and sets up symbolic links to ensure CUDA apps dynamically load this virtualised library when compiled with `nvcc .. cudart=shared`
- default cuda library is restoreable via `make restore-client-lib`.

2. Build the flyt control managers: `make install-cmgrs`
- `flyt-client-manager`(single instance per VM) runs on the client node.

3. Ensure `configs/client-mgr.toml` is set to appropriate values.
- [resource-manager]: address of cluster-manager node and its client listen port.
- [vcuda-client]:
- [ipc]:

4. Start the flyt-client-manager: `./bin/flyt-client-manager`

5. Build the benchmark test and run it: `cd synthetic_benchmark; make; bash run_benchE.sh`
- This runs a CUDA app on the VM whose runtime API calls are routed to the remote GPU cluster via the virtualised CUDA runtime library `cricket-client.so`.


## Build-And-Test Setup B
Build Flyt from source via `make` on a machine with all flyt dependencies installed. and share the `/bin`, `/configs` to the other nodes. Modify the `LD_LIBRARY_PATH` variable on client and cluster nodes to search the `/bin` folder for custom dynamically loaded libraries.



memcpy src:  0x7fffc4d86010-  0x7fffc6c0a810
memcpy dest: 0x7fffe6ec1000