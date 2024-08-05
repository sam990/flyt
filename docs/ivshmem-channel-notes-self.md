Key observations:
-----------------
- Frontend is single-threaded.
- Single ivshmem device per VM --> Single backend on the GPU host per VM.
- Multithreaded Flyt backend maintains a persistent TCP connection (for RPC) for each process in a VM.

1. Per CUDA process isolation of the shared-memory backend.
===========================================================
- Flyt front-end passes the data of multiple CUDA processes on a VM using the same shared-memory file.
- Implement per-process memory isolation in the flyt-frontend by paritioning the mmaped backend.
---> Flyt frontend maintains internal mappings of per-process shared-memory areas (similar to an OS, maybe simple pid to pages array?)
- Dynamic allocation of shared-memory areas on CUDA process creation (i.e. at every connection initiation by frontend?)
---> When a new connection is initiated, frontend will send an RPC packet to the backend containing
(i) integer pid of new host process.
(ii) starting virtual address (returned when shm region alloc'd by frontend) of the shared-memory region of the new process.
This <pid:VA> is stored in a map on the backend (or in a per-thread data structure) and used during D2H transfers to enable backend thread to write to the appropriate shm region.

Implementation
--------------
(see weensyOs process_setup for hints)
In the process init function:
1. init a page_table via init_process()

2. Implementing a faster `cudaMemcpy` (H2D and D2H) via Flyt shared-memory mode.
================================================================================
A Backend thread is created whenever a CUDA process is created, i.e. whenever frontend initiates a connection request to backend.
In shared-memory mode, the RPC channel is used for control signalling only.

- In H2D `cudaMemcpy_modified`, (all of this will likely be implemented in the modified runtime lib)
a. Frontend fills the allocated shm-region of the requesting CUDA process with user-space data to be copied.
b. Frontend creates and sends an RPC control packet containing: 
(i) integer index (interpreted by receiver as index into a shared-mem region) 
(ii) integer size (interpreted by receiver as size of data to be `cudaMemcpy`d)
(iii) pid of the requesting process (just to verify with the thread-local pid passed during initial setup)
c.Frontend sends the actual `cudaMemcpy` RPC command packet (backend now invokes the `cudaMemcpy` handler, shm metadata is available already.)

Note that the order of operations could also be (a), (c), (b)
- In this scenario, the `cudaMemcpy` handler could wait for an update to...? Not possible.


Since the backend-thread manages a single TCP connection with a process, the control metadata can be used 
by backend `cudaMemcpy` handler (along with initial setup metadata) to copy data from the correct location in the shared memory region to the flyt backend process' heap.
---> The original CUDA runtime API library `cudaMemcpy` is passed the newly created backend user-space pointer to perfrom the actual ()->GPU data transfer.
---> On success, an RPC success message is sent back to the requesting client.

- In D2H `cudaMemcpy`,
---> 

- For async calls
--->  

3. Frontend state Management.
=============================
- Maintain in-flight command name, remove when success message recd.
- Track shared memory allocations.









