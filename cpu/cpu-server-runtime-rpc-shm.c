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

void _svc_resp_do_notify(cricket_client *client) {
    // write to poll_s
    uint8_t *notif = (uint8_t *)client->ivshmem_ctx->shm_mmap + 2;
    //printf("Notif before: 0x%02X\n", *(notif));
    
    *notif = 1; // set poll_c
    //printf("poll_c set by server\n");
}

void print_neighbors(uint8_t *ptr, size_t num_bytes) {
    printf("Neighboring memory values:\n");
    for (size_t i = 0; i < num_bytes; ++i) {
        printf("Byte %zu: 0x%02X\n", i, *(ptr + i));
    }
}

void rpc_shm_svc_cuda_get_device_count_1(rpc_shm_header_t *rpc_hdr_svc, int_result *result, cricket_client *client) {
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
    if (((uintptr_t)(client->ivshmem_ctx->shm_mmap + resp_off) % alignof(int_result)) != 0) {
    printf("Error: Misaligned write for int_result at offset %zu\n", resp_off);
    // Handle misalignment, e.g., adjust r_d_off or handle the error
}

    // memcpy result to shm at response offset
    memcpy(client->ivshmem_ctx->shm_mmap + resp_off, result, sizeof(int_result));

    size_t num_bytes_to_print = 16;
    //print_neighbors((uint8_t *)client->ivshmem_ctx->shm_mmap + resp_off - num_bytes_to_print, num_bytes_to_print * 2 + 1);

    // update rpc_status
    rpc_hdr_svc->rpc_status = RPC_SHM_SUCCESS;

    // this is likely redundant.
    rpc_hdr_svc->rpc_response_desc.offset = resp_off;
    rpc_hdr_svc->rpc_response_desc.sz = sizeof(int_result);
    
    // notify
    _svc_resp_do_notify(client);

}


void rpc_shm_svc_cuda_get_device_properties_1(rpc_shm_header_t *rpc_hdr_svc, int_result *result, cricket_client *client) {
    _svc_resp_do_notify(client);
}