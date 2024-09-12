/*
 * FILE: cpu-client-ivshmem.h
 * --------------------------
 * Client-side ivshmem primitives.
*/
#ifndef __CPU_CLIENT_IVSHMEM_H__
#define __CPU_CLIENT_IVSHMEM_H__

#include <stddef.h>
#include "cpu_rpc_prot.h"
#include <stdint.h>

#define SHM_BE_PATH_SZ 128

#define SHM_OK 1
#define SHM_NONE 0

typedef struct __ivshmem_area {
    size_t max_size;
    size_t avail_size;             
    off_t avail_offset; // returns lowest un-written offset.
} _ivshmem_area;

// One per cuda process i.e.
// one per flyt libcudart instance
typedef struct __ivshmem_clnt_ctx {
    int shm_enabled;
    int pid;
    ivshmem_setup_desc svc_args;
    off_t shm_proc_start; // shm_mmap + shm_proc_start: lowest VA of the mmaped shm of this process.
    int shm_proc_size;
    void *shm_mmap;
    char shm_be_path[SHM_BE_PATH_SZ];

    _ivshmem_area read_from;  
    _ivshmem_area write_to;
} ivshmem_clnt_ctx;

extern ivshmem_clnt_ctx *ivshmem_ctx;

void init_ivshmem_clnt();
void init_ivshmem_areas_clnt(ivshmem_clnt_ctx *ctx);

char *_get_pci_path_clnt();

void _clnt_mgr_update_shm(int clnt_pid);
ivshmem_setup_desc _clnt_mgr_get_shm(int clnt_pid, char *shm_be_path);


uintptr_t shm_get_writeaddr_clnt(ivshmem_clnt_ctx *ctx);
uintptr_t shm_get_readaddr_clnt(ivshmem_clnt_ctx *ctx);

off_t shm_get_write_area_offset(size_t sz); // default: 0
off_t shm_get_read_area_offset(size_t sz); // default: 0

int check_shm_limits( _ivshmem_area *area, int size);

#endif