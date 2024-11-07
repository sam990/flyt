#include "cpu-client-ivshmem.h"
#include "cpu-client-runtime-rpc-shm.h"
#include "cpu_rpc_prot.h" // for return type
#include <string.h>
#include <stddef.h>
#include <assert.h>

#define RPC_SHM_INT 0
#define RPC_SHM_INT_64 1
#define RPC_SHM_DATA_PTR_64 2 // all struct args must be memcpied to data region
#define RPC_SHM_NO_DATA_PTR_64 3

void _clnt_req_do_notify() {
    // write to poll_s
    *((uint8_t *)ivshmem_ctx->shm_mmap + 1) = 1; // set poll_s
}

uint64_t _rpc_shm_get_offset_of_arg(int _idx) {
    uint64_t offset = offsetof(rpc_shm_header_t, rpc_args) + _idx * sizeof(struct rpc_shm_arg); // if 8 bytes in [0..(7-8)], [(8-9), (9-10) are next arg.]
    return offset;
}

uint64_t _rpc_shm_get_offset_of_arg_data(int _idx) {
;
}

uint64_t rpc_shm_clnt_get_response_status() {
    uint64_t _off = offsetof(rpc_shm_header_t, rpc_status);
    return *(uint8_t *)((uint8_t *)ivshmem_ctx->shm_mmap + _off);  
}

uint64_t rpc_shm_clnt_get_response_data_offset() {
;
}

uint64_t rpc_shm_clnt_get_response_data_sz() {
    ;
}

void print_neighbors(uint8_t *ptr, size_t num_bytes) {
    printf("Neighboring memory values:\n");
    for (size_t i = 0; i < num_bytes; ++i) {
        printf("Byte %zu: 0x%02X\n", i, *(ptr + i));
    }
}

void rpc_shm_clnt_put_request_and_notify(rpc_shm_header_t *rpc_hdr) {
    // memcpy the request metadata into shared memory.
    memcpy(ivshmem_ctx->shm_mmap, rpc_hdr, sizeof(rpc_shm_header_t));

    assert(*((uint8_t *)ivshmem_ctx->shm_mmap) == RPC_SHM_MAGIC_START);
    assert(*((uint8_t *)ivshmem_ctx->shm_mmap + 1) == 0);
    assert(*((uint8_t *)ivshmem_ctx->shm_mmap + 2) == 0);
    printf("pid byte: 0x%02X\n", *(uint32_t *)((uint8_t *)ivshmem_ctx->shm_mmap + 3));
    assert(*(uint32_t *)((uint8_t *)ivshmem_ctx->shm_mmap + 3) == 0xDEADBEEF);


    // printf("Byte before magic_end: 0x%02X\n", *((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) - 1));
    printf("Magic end byte: 0x%02X\n", *((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) - 1));
    // printf("Byte after magic_end: 0x%02X\n", *((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) - 1));

    size_t num_bytes_to_print = 16;
    printf("sizeof hdr: %d\n", sizeof(rpc_shm_header_t));
    //print_neighbors((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) - num_bytes_to_print, num_bytes_to_print * 2 + 1);
    
    assert(*((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) -1) == RPC_SHM_MAGIC_END);

    for (int _arg = 0; _arg < rpc_hdr->num_args; _arg++) {
        struct rpc_shm_arg *arg = &(rpc_hdr->rpc_args[_arg]);
        if (arg->arg_type == RPC_SHM_DATA_PTR_64) {
            // 1. Choose an offset (from start of memory) for the data
            // common data region.
            uint64_t d_off = sizeof(rpc_shm_header_t) + RPC_SHM_ARG_DATA_START - 1;
            
            arg->raw_info.d_off = d_off;
            
            // copy to shm
            uint64_t arg_off = _rpc_shm_get_offset_of_arg(_arg);
            *((uint64_t *)((uint8_t *)ivshmem_ctx->shm_mmap + arg_off)) = d_off;

            // 3. memcpy the arg data into shared memory.
            // issue: How much to copy? set by caller
            // issue: src: set by caller.
            // val would be a 64 bit uintptr
            memcpy((void *)((uint8_t *)ivshmem_ctx->shm_mmap + d_off), (void *)arg->raw_info.val, arg->arg_data_len);
        } else if (arg->arg_type == RPC_SHM_INT ||
            arg->arg_type == RPC_SHM_INT_64 ||
            arg->arg_type == RPC_SHM_NO_DATA_PTR_64) {   
                // do nothing
                continue;
        } else {
            printf("Invalid input argument type.\n");
        }

    }
    __sync_synchronize();

    // notify
    _clnt_req_do_notify();

}

// need a global pid 
uint64_t rpc_shm_clnt_cuda_get_device_count_1(int_result *res) {
    // create control struct
    rpc_shm_header_t *rpc_hdr = malloc(sizeof(rpc_shm_header_t));

    rpc_hdr->rpc_magic_start = RPC_SHM_MAGIC_START;
    rpc_hdr->rpc_magic_end = RPC_SHM_MAGIC_END;

    // Initialize the poll bytes
    rpc_hdr->poll_s = 0;
    rpc_hdr->poll_c = 0;
    rpc_hdr->pid = 0xDEADBEEF;
    rpc_hdr->rpc_status = RPC_SHM_FAILURE;
    rpc_hdr->rpc_cmd = CUDA_GET_DEVICE_COUNT;

    rpc_hdr->num_args = 0;
    memset(rpc_hdr->rpc_args, 0, sizeof(rpc_hdr->rpc_args));

    rpc_shm_clnt_put_request_and_notify(rpc_hdr); // memcpy to a fixed location in shm
    __sync_synchronize();

    free(rpc_hdr);
    printf("reached the condvar wait\n");

    // wait for cond_var
    printf("got_resp: %d\n", got_response);
    pthread_mutex_lock(&poll_mutex);
    // Wait in a loop to handle spurious wake-ups and missed signals
    while (!got_response) {
        pthread_cond_wait(&poll_cond_var, &poll_mutex);
    }
    got_response = 0;
    pthread_mutex_unlock(&poll_mutex);

    // read 
    int err = rpc_shm_clnt_get_response_status();
    uint64_t r_d_off = rpc_shm_clnt_get_response_data_offset(); // start offset of response data
    uint64_t r_d_sz = rpc_shm_clnt_get_response_data_sz();

    // get result
    if (err == RPC_SHM_FAILURE) {
        res->err = RPC_FAILED;
    } else {
        res->err = RPC_SUCCESS;
        memcpy(&(res->int_result_u.data), (void *)((uint8_t *)ivshmem_ctx->shm_mmap + r_d_off), r_d_sz); // this will be call-dependent.
    }
    return err;
}


uint64_t rpc_shm_clnt_cuda_get_device_properties_1(cuda_device_prop_result *res, int device) {
    rpc_shm_header_t *rpc_hdr = malloc(sizeof(rpc_shm_header_t));

    rpc_hdr->rpc_magic_start = RPC_SHM_MAGIC_START;
    rpc_hdr->rpc_magic_end = RPC_SHM_MAGIC_END;

    // Initialize the poll bytes
    rpc_hdr->poll_s = 0;
    rpc_hdr->poll_c = 0;

    // rpc_hdr->pid = 
    rpc_hdr->rpc_status = RPC_SHM_FAILURE;
    
    // cmd
    rpc_hdr->rpc_cmd = CUDA_GET_DEVICE_PROPERTIES;
    
    // set args
    rpc_hdr->num_args = 1;
    memset(rpc_hdr->rpc_args, 0, sizeof(rpc_hdr->rpc_args));
    rpc_hdr->rpc_args[0].arg_type = RPC_SHM_INT;
    rpc_hdr->rpc_args[0].raw_info.val = device; 

    rpc_shm_clnt_put_request_and_notify(rpc_hdr); // memcpy to a fixed location in shm
    __sync_synchronize();
    free(rpc_hdr);
}