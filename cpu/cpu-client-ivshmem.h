/*
 * FILE: cpu-client-ivshmem.h
 * --------------------------
 * Client-side ivshmem primitives.
 * Also define rpc over shm helpers.
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
    uint64_t avail_offset; // returns lowest un-written offset.
} _ivshmem_area;

// One per cuda process i.e.
// one per flyt libcudart instance
typedef struct __ivshmem_clnt_ctx {
    uint8_t shm_enabled;
    int pid;
    int clnt_mgr_mq;
    ivshmem_setup_desc svc_args;
    uint64_t shm_proc_start; // shm_mmap + shm_proc_start: lowest VA of the mmaped shm of this process.
    uint64_t shm_proc_size;
    void *shm_mmap;
    char shm_be_path[SHM_BE_PATH_SZ];

    _ivshmem_area read_from;  
    _ivshmem_area write_to;
} ivshmem_clnt_ctx;

extern ivshmem_clnt_ctx *ivshmem_ctx;

void init_ivshmem_clnt(int clnt_pid, char *shm_be_path, int clientd_mq_id);
void init_ivshmem_areas_clnt(ivshmem_clnt_ctx *ctx);

char *_get_pci_path_clnt();

ivshmem_setup_desc *_clnt_mgr_get_shm(int clnt_pid, int clientd_mq_id);
void clnt_mgr_free_shm(int clnt_pid, int clientd_mq_id);

uintptr_t shm_get_writeaddr_clnt(ivshmem_clnt_ctx *ctx);
uintptr_t shm_get_readaddr_clnt(ivshmem_clnt_ctx *ctx);

uint64_t shm_get_write_area_offset(size_t sz); // default: 0
uint64_t shm_get_read_area_offset(size_t sz); // default: 0

int check_shm_limits(_ivshmem_area *area, int size);


extern pthread_cond_t poll_cond_var;
extern pthread_mutex_t poll_mutex;
extern int got_response;

// rpc shm helpers
#define RPC_SHM_SUCCESS 0
#define RPC_SHM_FAILURE 1
#define RPC_SHM_MAGIC_START 0xD6 // 1 byte at beginnning of each RPC message. First byte of shm must always be = 0xD6
#define RPC_SHM_MAGIC_END 0xC5
#define RPC_SHM_ARG_DATA_START 8
#define RPC_SHM_MAX_ARG_LENGTH 128 // consistent with cudageterrorstring on server.

// 1 + 7 + 8 + 8 + 8 = 32 bytes/arg
struct rpc_shm_arg {
    uint8_t arg_type;
    uint8_t __pad3[7];
    struct {
        uint64_t val;
        uint64_t d_off;
    } raw_info;
    uint64_t arg_data_len;
}__attribute__((packed));

// 16 bytes /response meta
typedef struct rpc_shm_response {
    uint64_t sz; // set by server
    uint64_t offset; // read by client
}__attribute__((packed)) rpc_shm_response_t ;

// 1 + 1 + 1 + 4 + 1 + 4 + 1 + 3 + 32 *16 + 16 + 1 = 545 bytes.
typedef struct rpc_shm_header {
    uint8_t rpc_magic_start;
    uint8_t poll_s; // (R:Server, W: Client) On write by client, client sets it to 1, server reads then resets to 0
    uint8_t poll_c; // (R: Client, W: Server) On write by server, server sets it to 1, client reads then resets to 0
    uint32_t pid;
    uint8_t rpc_status; // RPC_SUCCESS/FAILURE

    uint32_t rpc_cmd;
    uint8_t num_args;
    uint8_t __pad2[3];

    struct rpc_shm_arg rpc_args[16]; // max 16 rpc args allowed.

    rpc_shm_response_t rpc_response;
    uint8_t rpc_magic_end;
}__attribute__((packed)) rpc_shm_header_t ;

void rpc_shm_clnt_put_request_and_notify(rpc_shm_header_t *rpc_hdr); // copy rpc control header to shm, update poll bit
uint64_t rpc_shm_clnt_get_response_status(); // to access status
uint64_t rpc_shm_clnt_get_response_data_offset();


#endif