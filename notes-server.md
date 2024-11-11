# Shared Memory RPCs
Idea: Server reads RPC commands, arguments and optional data from the ivshmem area.
These values can be referenced by simple offsets.
```
// when poll bit = 1, condvar signal from poll thread.
// on client, 
0 <server POLL bit (R:Server, W: Client)> 0: On write by client, client sets it to 1 (server reads then resets to 0)
1 <client POLL bit (R: Client, W: Server)> 0:  On write by server, server sets it to 1 (client reads then resets to 0)
2 
X
X
X
X
X
-
<arg1>: |arg_type|  arg value  | (reader of shm does *(int *)(mmap_base_addr + arg_1_offset + 1)) 
-
<arg2>
-
<arg3>
-
<arg4>
-
<arg5>
-

the poll thread signals a condition variable
eg:
// on client
cudagetDevice()
if (shm)
    do_shm_rpc_write(__cuda_call_type, *args) // populate shm area with appropriate metadata
    condvar_wait(rpc_condvar) // block till poll thread signals

    do_shm_rpc_read() // read

// on server



INVARIANTS
- Single polling thread per process that loads the library (thread created in constructor.)
- The role of a polling thread is to signal a condvar
