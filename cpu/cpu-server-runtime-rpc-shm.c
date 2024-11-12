/**
 *
 * Note: Return types are are dependent on
 * tiRPC return types.
 */
#include "cpu-server-runtime-rpc-shm.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>
#include <dlfcn.h>
#include <cuda_profiler_api.h>
#include <stdalign.h>
#include <sys/mman.h>
#include <emmintrin.h>

// inline void clflush(volatile void *p)
// {
//     asm volatile ("clflush (%0)" :: "r"(p));
// }

void print_neighbors(uint8_t *ptr, size_t num_bytes) {
    printf("Neighboring memory values:\n");
    for (size_t i = 0; i < num_bytes; ++i) {
        printf("Byte %zu: 0x%02X\n", i, *(ptr + i));
    }
}

void _svc_resp_do_notify(cricket_client *client) {
    // write to poll_s
    //uint8_t *notif = (uint8_t *)client->ivshmem_ctx->shm_mmap + 2;
    //printf("Notif before: 0x%02X\n", *(notif));
    uint8_t value = 1;
    memcpy((uint8_t *)client->ivshmem_ctx->shm_mmap + 2, &value, sizeof(value)); // set poll_c
    // printf("poll_c set by server\n");

    // Perform cache flush to ensure visibility
    // clflush((uint8_t *)client->ivshmem_ctx->shm_mmap + 2);
    __sync_synchronize();

    // printf("poll_c val set to by svc: %d\nneighbours after:\n", *((uint8_t *)client->ivshmem_ctx->shm_mmap + 2));

    // int num_bytes_to_print = 8;
    // print_neighbors((uint8_t *)client->ivshmem_ctx->shm_mmap + 1, num_bytes_to_print * 2 + 1);

}

void rpc_shm_svc_cuda_get_device_count_1(volatile rpc_shm_header_t *rpc_hdr_svc, int_result *result, cricket_client *client) {
    // no args
    // do result struct, copy into shm, update
    // result offset, notify.
    // (cant response struct also be memcpied by client at init??)
    // do this later.
    result->int_result_u.data = 1;
    result->err = cudaSuccess;

    // get offset of response data (to fill response_desc)
    // response will always be in the data section.
    // response may be a simple struct, or even data.
    uint64_t resp_off = sizeof(rpc_shm_header_t) + RPC_SHM_ARG_DATA_START - 1;
//     if (((uintptr_t)(client->ivshmem_ctx->shm_mmap + resp_off) % alignof(int_result)) != 0) {
//     printf("Error: Misaligned write for int_result at offset %zu\n", resp_off);
//     // Handle misalignment, e.g., adjust r_d_off or handle the error
// }

    // memcpy result to shm at response offset
    memcpy(client->ivshmem_ctx->shm_mmap + resp_off, result, sizeof(int_result));

    size_t num_bytes_to_print = 8;
    //print_neighbors((uint8_t *)client->ivshmem_ctx->shm_mmap + resp_off - num_bytes_to_print, num_bytes_to_print * 2 + 1);

    // update rpc_status
    rpc_hdr_svc->rpc_status = RPC_SHM_SUCCESS;
    __sync_synchronize();

    // asm volatile("clflush (%0)" :: "r" (rpc_hdr_svc));

    // msync((void *)rpc_hdr_svc, sizeof(rpc_shm_header_t), MS_SYNC);

    // __sync_synchronize();

    // 
    
    // printf("status offset  %d\n", offsetof(rpc_shm_header_t, rpc_status));

    // print_neighbors((uint8_t *)client->ivshmem_ctx->shm_mmap + 1, num_bytes_to_print * 2 + 1);


    // this is likely redundant.
    rpc_hdr_svc->rpc_response_desc.offset = resp_off;
    rpc_hdr_svc->rpc_response_desc.sz = sizeof(int_result);
    
    // notify
    _svc_resp_do_notify(client);

    // _mm_mfence();
    //msync((void *)rpc_hdr_svc, sizeof(rpc_shm_header_t), MS_SYNC);

}


void rpc_shm_svc_cuda_get_device_properties_1(volatile rpc_shm_header_t *rpc_hdr_svc, int_result *result, cricket_client *client) {
    _svc_resp_do_notify(client);
}