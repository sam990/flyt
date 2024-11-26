#include "cpu-client-ivshmem.h"
#include "cpu-client-runtime-rpc-shm.h"
#include "cpu_rpc_prot.h" // for return type
#include <string.h>
#include <stddef.h>
#include <assert.h>
#include <stdalign.h>
#include <unistd.h>
#include <emmintrin.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define RPC_SHM_INT 0
#define RPC_SHM_INT_64 1
#define RPC_SHM_DATA_PTR_64 2 // all struct args must be memcpied to data region
#define RPC_SHM_NO_DATA_PTR_64 3

inline void clflush(volatile void *p)
{
    asm volatile ("clflush (%0)" :: "r"(p));
}


void _clnt_req_do_notify(int _notify_at) {
    // write to poll_s
    *((uint8_t *)ivshmem_ctx->shm_mmap + _notify_at) = 1; // set poll_s
    // clflush((uint8_t *)ivshmem_ctx->shm_mmap + 1);
}

uint64_t _rpc_shm_get_offset_of_arg(int _idx) {
    uint64_t offset = offsetof(rpc_shm_header_t, rpc_args) + _idx * sizeof(struct rpc_shm_arg); // if 8 bytes in [0..(7-8)], [(8-9), (9-10) are next arg.]
    return offset;
}

uint64_t _rpc_shm_get_offset_of_arg_data(int _idx) {
;
}

uint8_t rpc_shm_clnt_get_response_status() {
    uint64_t _off = offsetof(rpc_shm_header_t, rpc_status);
    return *(uint8_t *)((uint8_t *)ivshmem_ctx->shm_mmap + _off);  
}

uint64_t rpc_shm_clnt_get_response_data_offset() {
    return sizeof(rpc_shm_header_t) + RPC_SHM_ARG_DATA_START - 1;
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

rpc_shm_header_t *init_rpc_shm_header(int _cuda_api_call, int num_args) {
    rpc_shm_header_t *rpc_hdr = malloc(sizeof(rpc_shm_header_t));

    rpc_hdr->rpc_magic_start = RPC_SHM_MAGIC_START;
    rpc_hdr->rpc_magic_end = RPC_SHM_MAGIC_END;

    // Initialize the poll bytes
    rpc_hdr->poll_s = 0;
    rpc_hdr->poll_c = 0;

    rpc_hdr->pid = 0xDEADBEEF;
    rpc_hdr->rpc_status = RPC_SHM_NONE; // set to success by server.

    rpc_hdr->rpc_cmd = _cuda_api_call;

    rpc_hdr->num_args = num_args;
    memset(rpc_hdr->rpc_args, 0, sizeof(rpc_hdr->rpc_args));

    return rpc_hdr;
}

void rpc_shm_clnt_put_request_and_notify(volatile rpc_shm_header_t *rpc_hdr, int _notify_at) {
    // memcpy the request metadata into shared memory.
    memcpy((void *)ivshmem_ctx->shm_mmap, rpc_hdr, sizeof(rpc_shm_header_t));
    __sync_synchronize();
    msync(ivshmem_ctx->shm_mmap, sizeof(rpc_shm_header_t), MS_SYNC);

    uintptr_t _addr = (uintptr_t)ivshmem_ctx->shm_mmap;
    uintptr_t _end = _addr + sizeof(rpc_shm_header_t);

    assert(*((uint8_t *)ivshmem_ctx->shm_mmap) == RPC_SHM_MAGIC_START);
    assert(*((uint8_t *)ivshmem_ctx->shm_mmap + 1) == 0);
    assert(*((uint8_t *)ivshmem_ctx->shm_mmap + 2) == 0);
    printf("pid byte: 0x%02X\n", *(uint32_t *)((uint8_t *)ivshmem_ctx->shm_mmap + 3));
    assert(*(uint32_t *)((uint8_t *)ivshmem_ctx->shm_mmap + 3) == 0xDEADBEEF);

    size_t num_bytes_to_print = 4;
    print_neighbors((uint8_t *)ivshmem_ctx->shm_mmap + offsetof(rpc_shm_header_t, rpc_status) - num_bytes_to_print,num_bytes_to_print * 2 + 1);
    assert(*((uint8_t *)ivshmem_ctx->shm_mmap + offsetof(rpc_shm_header_t, rpc_status)) == RPC_SHM_NONE);


    printf("Byte before magic_end: 0x%02X\n", *((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) - 1));
    printf("Magic end byte: 0x%02X\n", *((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t)));
    printf("Byte after magic_end: 0x%02X\n", *((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) + 1));
    printf("sizeof hdr: %d\n", sizeof(rpc_shm_header_t));
    print_neighbors((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) - num_bytes_to_print, num_bytes_to_print * 2 + 1);
    
    assert(*((uint8_t *)ivshmem_ctx->shm_mmap + sizeof(rpc_shm_header_t) -1) == RPC_SHM_MAGIC_END);

    for (int _arg = 0; _arg < rpc_hdr->num_args; _arg++) {
        volatile struct rpc_shm_arg *arg = &(rpc_hdr->rpc_args[_arg]);
        if (arg->arg_type == RPC_SHM_DATA_PTR_64) {
            // 1. Choose an offset (from start of memory) for the data
            // common data region.
            // note this is same as response section!
            // INVARIANT: argument is always consumed before a response is ready.
            uint64_t d_off = sizeof(rpc_shm_header_t) + RPC_SHM_ARG_DATA_START - 1; 
            
            arg->raw_info.d_off = d_off; // location of deviceprop in memory.
            
            // copy offset of arg data to shm
            uint64_t arg_off = _rpc_shm_get_offset_of_arg(_arg); // where to place the arg
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
    free((void *)rpc_hdr);

    // notify
    _clnt_req_do_notify(_notify_at);

}

#define TIMEOUT_MS 1000
void busy_poll_and_wait_for_response(int _at_byte) {
    struct timeval start, current;
    gettimeofday(&start, NULL);  // Record the start time

    for (;;) {
        // Calculate elapsed time
        gettimeofday(&current, NULL);
        long elapsed_ms = (current.tv_sec - start.tv_sec) * 1000 + 
                          (current.tv_usec - start.tv_usec) / 1000;

        if (elapsed_ms >= TIMEOUT_MS) {
            printf("Timeout reached\n");
            break;
        }

        // Poll the memory-mapped byte
        if (*((uint8_t *)ivshmem_ctx->shm_mmap + _at_byte) == 1) {
            // Clear the byte and exit the loop
            *((uint8_t *)ivshmem_ctx->shm_mmap + _at_byte) = 0;
            return;
        }

        usleep(1);
    }
}

// uint64_t rpc_shm_clnt_cuda_choose_device_1(const struct cudaDeviceProp *prop, int_result *result) {
//     volatile rpc_shm_header_t *rpc_hdr = init_rpc_shm_header(CUDA_CHOOSE_DEVICE, 1);
//     rpc_hdr->rpc_args[0].arg_type = RPC_SHM_DATA_PTR_64;
//     rpc_hdr->rpc_args[0].arg_data_len = sizeof(struct cudaDeviceProp);

//     rpc_shm_clnt_put_request_and_notify(rpc_hdr, POLL_S);

//     busy_poll_and_wait_for_response(POLL_C);

//     uint64_t r_d_off = rpc_shm_clnt_get_response_data_offset();

//     msync(ivshmem_ctx->shm_mmap + r_d_off, sizeof(int_result), MS_INVALIDATE);

//     *result = *((int_result *)(ivshmem_ctx->shm_mmap + r_d_off)); // will be copied to a region on stack.
    
//     printf("result data from svc: %d\n", result->int_result_u.data);
//     printf("error code: %x\n", result->err);

//     int err = rpc_shm_clnt_get_response_status();

//     // print_neighbors((uint8_t *)ivshmem_ctx->shm_mmap + offsetof(rpc_shm_header_t, rpc_status) - num_bytes_to_print, num_bytes_to_print * 2 + 1);
//     if (err == RPC_SHM_FAILURE) {
//         printf("rpc shm failure written by server\n");
//         result->err = RPC_FAILED; // this will change the actual shm
//     } else if (err == RPC_SHM_SUCCESS) {
//         // printf("rpc shm success written by server\n");
//         result->err = RPC_SUCCESS;
//         //memcpy(&(res->int_result_u.data), (void *)((uint8_t *)ivshmem_ctx->shm_mmap + r_d_off), r_d_sz); // this will be call-dependent.
//     } 
    
//     return err;

// }


// need a global pid 
uint64_t rpc_shm_clnt_cuda_get_device_count_1(int_result *res) {
    // create control struct
    volatile rpc_shm_header_t *rpc_hdr = init_rpc_shm_header(CUDA_GET_DEVICE_COUNT, 0);

    rpc_shm_clnt_put_request_and_notify(rpc_hdr, POLL_S); // memcpy to a fixed location in shm

    busy_poll_and_wait_for_response(POLL_C); // add a timeout

    // read 
    uint64_t r_d_off = rpc_shm_clnt_get_response_data_offset(); // start offset of response data
    
    msync(ivshmem_ctx->shm_mmap + r_d_off, sizeof(int_result), MS_INVALIDATE);

    *res = *((int_result *)(ivshmem_ctx->shm_mmap + r_d_off)); // will be copied to a region on stack.
    printf("result data from svc: %d\n", res->int_result_u.data);

    printf("error code: %x\n", res->err);
    int num_bytes_to_print = 8;
    print_neighbors((uint8_t *)ivshmem_ctx->shm_mmap + r_d_off - num_bytes_to_print, num_bytes_to_print * 2 + 1);

    // get result
    int err = rpc_shm_clnt_get_response_status();

    // print_neighbors((uint8_t *)ivshmem_ctx->shm_mmap + offsetof(rpc_shm_header_t, rpc_status) - num_bytes_to_print, num_bytes_to_print * 2 + 1);
    if (err == RPC_SHM_FAILURE) {
        printf("rpc shm failure written by server\n");
        res->err = RPC_FAILED; // this will change the actual shm
    } else if (err == RPC_SHM_SUCCESS) {
        // printf("rpc shm success written by server\n");
        res->err = RPC_SUCCESS;
        //memcpy(&(res->int_result_u.data), (void *)((uint8_t *)ivshmem_ctx->shm_mmap + r_d_off), r_d_sz); // this will be call-dependent.
    } 
    
    return err;
}

uint64_t rpc_shm_clnt_cuda_get_device_properties_1(int device, cuda_device_prop_result *res) {

    rpc_shm_header_t *rpc_hdr = init_rpc_shm_header(CUDA_GET_DEVICE_PROPERTIES, 1); // issue: args need to be indep
    
    // set arg types.
    // I need to know whether a given pointer is a dataptr or 
    // regular ptr.
    rpc_hdr->rpc_args[0].arg_type = RPC_SHM_INT;
    rpc_hdr->rpc_args[0].raw_info.val = device; // if dataptr, offset of data.

    rpc_shm_clnt_put_request_and_notify(rpc_hdr, POLL_S); // memcpy to a fixed location in shm

    busy_poll_and_wait_for_response(POLL_C); // add a timeout

    uint64_t r_d_off = rpc_shm_clnt_get_response_data_offset(); // start offset of response data

    *res = *((cuda_device_prop_result *)(ivshmem_ctx->shm_mmap + r_d_off)); 

    int err = rpc_shm_clnt_get_response_status(); // RPC success/failure

    if (err == RPC_SHM_FAILURE) {
        printf("rpc shm failure written by server\n");
        res->err = RPC_FAILED; // this will change the actual shm
    } else if (err == RPC_SHM_SUCCESS) {
        // printf("rpc shm success written by server\n");
        res->err = RPC_SUCCESS;
    } 

    return err;
    
}