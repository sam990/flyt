/**
 * FILE: cpu-server-ivshmem.h
 * --------------------------
 * Defines primitives for ivshmem support
 * on the flyt-rpc-server. 
 * Answers the question "Where to write to/read from?"
 * Uses the Cricket list API
 */
#ifndef __CPU_SERVER_IVSHMEM_H__
#define __CPU_SERVER_IVSHMEM_H__

#include <stdint.h>
#include <stddef.h>
#include "cpu_rpc_prot.h"

#define SHM_BE_PATH_SZ 128

typedef struct __ivshmem_area {
    size_t max_size;
    size_t avail_size;             
    uint64_t avail_offset; // returns lowest un-written offset.
} _ivshmem_area;

// goes in cricket_client
typedef struct __ivshmem_svc_ctx {
    int pid;
    uint8_t shm_enabled;
    uint64_t shm_proc_start;
    uint64_t shm_proc_size;
    void *shm_mmap; // &shm_mmap[read_area.offset] = src mem_ptr passed to the cuda_memcpy_svc.
    char shm_be_path[SHM_BE_PATH_SZ];

    _ivshmem_area read_from;  
    _ivshmem_area write_to; 
} ivshmem_svc_ctx;

ivshmem_svc_ctx *init_ivshmem_svc(ivshmem_setup_desc args_from_clnt);

void init_ivshmem_areas_svc(ivshmem_svc_ctx *ctx);


// memcpy helpers.
// in D2H: svc dst_addr = write_getaddr() + r_off_clnt
// in H2D: svc src_addr = read_getaddr() + w_off_clnt
uintptr_t shm_get_readaddr_svc(ivshmem_svc_ctx *_ctx); // returns VA of start of mmaped region corresp to ivshmem_area read_from (shm_mmap + PROC_READ_OFFSET_SVC)
uintptr_t shm_get_writeaddr_svc(ivshmem_svc_ctx *_ctx); // returns VA of start of mmaped region corresp to ivshmem_area write_to (shm_mmap + PROC_WRITE_OFFSET_SVC)

#endif

