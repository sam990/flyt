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
#include "log.h"
#include "api-recorder.h"
#include "gsched.h"
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

void print_rpc_arg(const struct rpc_shm_arg *arg) {
    printf("  arg_type: %u\n", arg->arg_type);
    printf("  raw_info.val: %llu\n", arg->raw_info.val);
    printf("  raw_info.d_off: %llu\n", arg->raw_info.d_off);
    printf("  arg_data_len: %llu\n", arg->arg_data_len);
}

// Function to print rpc_shm_response_t
void print_rpc_response_desc(const rpc_shm_response_t *response) {
    printf("  rpc_response_desc.sz: %llu\n", response->sz);
    printf("  rpc_response_desc.offset: %llu\n", response->offset);
}

// Function to print rpc_shm_header_t
void print_rpc_header(const rpc_shm_header_t *rpc_hdr_svc) {
    printf("rpc_shm_header_t details:\n");

    printf("rpc_magic_start: %x\n", rpc_hdr_svc->rpc_magic_start);
    printf("poll_s: %u\n", rpc_hdr_svc->poll_s);
    printf("poll_c: %u\n", rpc_hdr_svc->poll_c);
    printf("pid: %x\n", rpc_hdr_svc->pid);
    printf("rpc_status: %u\n", rpc_hdr_svc->rpc_status);
    printf("rpc_cmd: %u\n", rpc_hdr_svc->rpc_cmd);
    printf("num_args: %u\n", rpc_hdr_svc->num_args);

    // Print each rpc_shm_arg (max 16 args)
    for (int i = 0; i < rpc_hdr_svc->num_args; i++) {
        printf("rpc_args[%d]:\n", i);
        print_rpc_arg(&rpc_hdr_svc->rpc_args[i]);
    }

    // Print rpc_response_desc
    print_rpc_response_desc(&rpc_hdr_svc->rpc_response_desc);

    printf("rpc_magic_end: %u\n", rpc_hdr_svc->rpc_magic_end);
}

void rpc_shm_cuda_choose_device_1_svc(volatile rpc_shm_header_t *rpc_hdr_svc, int_result *result, cricket_client *client) {
    uint8_t num_args = *((uint8_t *)client->ivshmem_ctx->shm_mmap + offsetof(rpc_shm_header_t, num_args));
    if (num_args !=1) {
        printf("error num args\n");
    }

    int prop_data_len = rpc_hdr_svc->rpc_args->arg_data_len;

    if (prop_data_len != sizeof(struct cudaDeviceProp)) {
        LOGE(LOG_ERROR, "Received wrong amount of data: expected %zu but got %zu", sizeof(struct cudaDeviceProp), prop_data_len);
        return 0;
    }
    
    uint64_t prop_data_off = rpc_hdr_svc->rpc_args[0].raw_info.d_off;
    struct cudaDeviceProp *cudaProp = (struct cudaDeviceProp *)((uint8_t *)client->ivshmem_ctx->shm_mmap + prop_data_off);

    RECORD_API(struct cudaDeviceProp);
    RECORD_SINGLE_ARG(*cudaProp);

    GSCHED_RETAIN;
    result->err = cudaChooseDevice(&result->int_result_u.data, cudaProp);
    GSCHED_RELEASE;

    RECORD_RESULT(integer, result->int_result_u.data);

    // get response offset
    uint64_t resp_off = sizeof(rpc_shm_header_t) + RPC_SHM_ARG_DATA_START - 1;

    // copy results to shm
    memcpy(client->ivshmem_ctx->shm_mmap + resp_off, result, sizeof(cuda_device_prop_result));   

    // indicate success
    rpc_hdr_svc->rpc_status = RPC_SHM_SUCCESS;
    __sync_synchronize(); 
    msync((void *)rpc_hdr_svc, sizeof(rpc_shm_header_t), MS_SYNC);

    _svc_resp_do_notify(client);
}

void rpc_shm_svc_cuda_get_device_count_1(volatile rpc_shm_header_t *rpc_hdr_svc, int_result *result, cricket_client *client) {
    // no args
    // do result struct, copy into shm, update
    // result offset, notify.
    // (cant response struct also be memcpied by client at init??)
    // do this later.
    result->int_result_u.data = 1;
    result->err = cudaSuccess;
    // print_rpc_header(rpc_hdr_svc);

    // get offset of response data (to fill response_desc)
    // response will always be in the data section.
    // response may be a simple struct, or even data.
    uint64_t resp_off = sizeof(rpc_shm_header_t) + RPC_SHM_ARG_DATA_START - 1;
 
    // memcpy result to shm at response offset
    memcpy(client->ivshmem_ctx->shm_mmap + resp_off, result, sizeof(int_result));

    size_t num_bytes_to_print = 8;
    print_neighbors((uint8_t *)client->ivshmem_ctx->shm_mmap + resp_off - num_bytes_to_print, num_bytes_to_print * 2 + 1);

    // update rpc_status
    rpc_hdr_svc->rpc_status = RPC_SHM_SUCCESS;
    __sync_synchronize();

    msync((void *)rpc_hdr_svc, sizeof(rpc_shm_header_t), MS_SYNC);

    // printf("status offset  %d\n", offsetof(rpc_shm_header_t, rpc_status));

    // print_neighbors((uint8_t *)client->ivshmem_ctx->shm_mmap + 1, num_bytes_to_print * 2 + 1);


    // // this is likely redundant.
    // rpc_hdr_svc->rpc_response_desc.offset = resp_off;
    // rpc_hdr_svc->rpc_response_desc.sz = sizeof(int_result);
    
    // notify
    _svc_resp_do_notify(client);

}


void rpc_shm_svc_cuda_get_device_properties_1(volatile rpc_shm_header_t *rpc_hdr_svc, cuda_device_prop_result *result, cricket_client *client) {

    // num args
    uint8_t num_args = *((uint8_t *)client->ivshmem_ctx->shm_mmap + offsetof(rpc_shm_header_t, num_args));
    if (num_args !=1) {
        printf("error num args\n");
    }

    LOGE(LOG_DEBUG, "cudaGetDeviceProperties");
    if (sizeof(result->cuda_device_prop_result_u.data) != sizeof(struct cudaDeviceProp)) {
        LOGE(LOG_ERROR, "cuda_device_prop_result size mismatch, result %d prop %d", sizeof(result->cuda_device_prop_result_u.data), sizeof(struct cudaDeviceProp));
        return NULL;
    }

    // uint8_t device = get_arg(arg_type, at_index);
    uint8_t device = rpc_hdr_svc->rpc_args[0].raw_info.val;
    result->err = cudaGetDeviceProperties((void*)result->cuda_device_prop_result_u.data, device);

    // get response offset
    uint64_t resp_off = sizeof(rpc_shm_header_t) + RPC_SHM_ARG_DATA_START - 1;

    // copy results to shm
    memcpy(client->ivshmem_ctx->shm_mmap + resp_off, result, sizeof(cuda_device_prop_result));   

    // indicate success
    rpc_hdr_svc->rpc_status = RPC_SHM_SUCCESS;
    __sync_synchronize(); 
    msync((void *)rpc_hdr_svc, sizeof(rpc_shm_header_t), MS_SYNC);

    // notify
    _svc_resp_do_notify(client);

}