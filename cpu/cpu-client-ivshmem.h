/*
 * FILE: cpu-client-ivshmem.h
 * --------------------------
 * Client-side ivshmem primitives.
*/
#ifndef __CPU_CLIENT_IVSHMEM_H__
#define __CPU_CLIENT_IVSHMEM_H__

#include <stddef.h>
#include "cpu_rpc_prot.h"

#define SHM_BE_PATH_SZ 128

#define SHM_OK 1
#define SHM_NONE 0

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
} ivshmem_clnt_ctx;

extern ivshmem_clnt_ctx *ivshmem_ctx;

// mmaps the correct offset of the pci BAR.
// - shm_proc_start is obtained by requesting the client manager via message queue.
// - client manager returns lowest un-reserved offset.
// - after mmap(), send the client manager the size that was just mmaped to sync. client manager state for next process.
// update clnt_ctx
void init_ivshmem_clnt();
char *_get_pci_path_clnt();
void *shm_get_addr_clnt(); // start of mmaped shm region for this clnt.

void _clnt_mgr_update_shm(int clnt_pid);
ivshmem_setup_desc _clnt_mgr_get_shm(int clnt_pid, char *shm_be_path);

off_t shm_get_write_offset();
off_t shm_get_read_offset();

void check_shm_limits();

#endif