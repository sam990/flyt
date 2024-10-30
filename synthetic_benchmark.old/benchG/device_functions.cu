#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <cuda_runtime.h>

#define THREAD_PER_CLIENT 256
#define NUM_ITERATIONS 5000

#define HANDLE_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static size_t max_batch_size = 0;

static int post_data_bytes = 0;
static uint8_t *dmem;
static uint8_t *response_data_device;
static uint8_t *response_data_host;


__global__ void add(long iterations, uint8_t *d_a, uint8_t *d_b, uint8_t *d_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t i = (blockIdx.x * blockDim.x + threadIdx.x) & 0xFF;
    uint8_t b = i + 32;


    for (long k = 0; k < iterations; k++) {

        for (long j = 0; j < iterations; j++) {
            i += b;
            i -= b;
            i *= b;
        }
    }
    
    d_c[idx] += 2 * i + d_b[idx];
    // d_a[idx] += i + d_b[idx];
}

extern "C" void executeFunc(uint8_t* data, int* responses, int num_requests) {

    HANDLE_ERROR(cudaMemcpy(dmem, data, num_requests * post_data_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemset(response_data_device, 0, num_requests * post_data_bytes));

    int total_thread = num_requests * THREAD_PER_CLIENT;

    long long num_blocks = total_thread >> 5;

    add<<<num_blocks, 32>>>(NUM_ITERATIONS, dmem, dmem, response_data_device);

    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(response_data_host, response_data_device, num_requests * post_data_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_requests; i++) {
        responses[i] = 0;

        // printf("response: %d\n", i);

        for (int j = 0; j < post_data_bytes; j++) {
            responses[i] += response_data_host[i * post_data_bytes + j];
            // printf("%d ", response_data_host[i * post_data_bytes + j]);
        }
        // printf("\n");
    }

    return;
}


extern "C" int init_device_vars(size_t __post_data_bytes, size_t __max_batch_size) {
    post_data_bytes = __post_data_bytes;
    max_batch_size = __max_batch_size;
    HANDLE_ERROR(cudaMalloc((void**)&dmem, max_batch_size * post_data_bytes));
    HANDLE_ERROR(cudaMalloc((void**)&response_data_device, max_batch_size * post_data_bytes));
    response_data_host = (uint8_t*)malloc(max_batch_size * post_data_bytes);
    return 0;
}