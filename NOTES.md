Observations
-------------
- client (frontend) and server (backend) code is in `/cpu`
- `cpu-client-driver.c` contains the modified cuda runtime api implementations (i.e. the rpc calls)
- The main Makefile generates rpc protocol headers from `cpu_rpc_protocol.x`
- control-managers/configs.rs require path changes based on your local machine (see README)
- `flytctl` on a client VM queries the control manager daemon which is written in rust. (see ./flytctl help)
- From `Makefile`: 
1. Two daemons: One on flyt frontend (client-side) and one on flyt backend (server-side). The client-side daemon is `flyt-client-manager`, server-side daemon is `flyt-node-manager`. 
2. On flyt backend, the `cricket-rpc-server` is built for the flyt backend (which is now multithreaded) that performs the role of calling true cuda runtime api commands.
3. On flyt frontend, the client code in `cpu/` is used to build the `cricket-client.so` shared lib. This code implements the modified cuda library i.e. RPC to flyt backend and connection init.
- See $(SRC_CLIENT) in the cpu makefile. Notice that the client driver cpu-client-driver.c is built seperately.

- `server-exe` is the flyt backend server entry point.

In rust code, (See `Cargo.toml` for src-> bin mappings)
- `virt_server_program` is the cricket-rpc-server. 
- Rust "servernode-daemon" ==> flyt-node-manager binary.
---> servernode-daemon (flyt-node-manager) tries to connect to server (i.e cricket-rpc server) and resource manager.

- Rust client-manager-daemon ==> flyt-client-manager
---> Runs on client VM, listens for connections.

- Rust cluster-manager ==> flyt-cluster-manager, flytctl
---> Creates a flyt-frontend-socket i.e. UDS

Setup
-----
RPC-only: Several VMs running on a physical machine with no GPU. CUDA processes send RPC commands to a cluster of GPUs. A GPU cluster has a single public endpoint which receives messages from VMs.

Shared-mem: Several VMs running on a physical Machine with a GPU. CUDA processes running on VM send commands via an RPC channel to an end-point on 
the host, MPS-enabled machine 





Build/Run procedure
-------------------
- The machine that runs the flyt backend needs to have an MPS-enabled GPU. Note that MPS works on turing (GTX).
- `make` builds all NVIDIA GPU tests from the public repo. Can ignore build while testing modifications.
- `make install cpu-server`: Install the flyt backend executables on GPU host.
- `make install cpu-client`: Install the flyt frontend executables on the VM running the CUDA process.

vm to run clients: vm@10.129.26.124, pw: 1234