#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>


#define HANDLE_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


__global__ void add(long iterations, long *d_a, long *d_b, long *d_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x + threadIdx.x + 32;


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

void *get_random_array(long long size) {
    char *arr = (char*)malloc(size);

    if (arr == NULL) {
        fprintf(stderr, "Unable to allocate array on cpu\n");
        return NULL;
    }

    for (long long i = 0; i < size; i++) {
        arr[i] = (char)(rand() % CHAR_MAX);
    }
    return arr;
}



int main(int argc, char* argv[]) {

    srand(0x112);

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <mem-size(KB)> <num-threads> <num-iterations>\n", argv[0]);
        return -1;
    }

    long long mem_size = strtoll(argv[1], NULL, 10);
    long long num_threads = strtoll(argv[2], NULL, 10);
    long long num_iterations = strtoll(argv[3], NULL, 10);


    long long mem_bytes = mem_size << 10;
    
    if ((num_threads * 8) > mem_bytes) {
        fprintf(stderr, "Too many threads\n");
        return 1;
    }

    long *h_a = (long *)get_random_array(mem_bytes);

    long *h_x = (long *)calloc(mem_bytes, 1);
    long *h_y = (long *)calloc(mem_bytes, 1);
    long *h_z = (long *)calloc(mem_bytes, 1);

    long *d_a;
    long *d_b;
    long *d_c;

    HANDLE_ERROR(cudaDeviceSynchronize());
    
    timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    HANDLE_ERROR(cudaMalloc(&d_a, mem_bytes));
    HANDLE_ERROR(cudaMalloc(&d_b, mem_bytes));
    HANDLE_ERROR(cudaMalloc(&d_c, mem_bytes));

    HANDLE_ERROR(cudaMemset(d_c, 0, mem_bytes));
    HANDLE_ERROR(cudaMemcpy(d_a, h_a, mem_bytes, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b, h_a, mem_bytes, cudaMemcpyHostToDevice));

    cudaFuncAttributes attr;
    cudaError_t retuAttr = cudaFuncGetAttributes(&attr, add);

    printf("attr return val: %d\n", retuAttr);
    printf("Function Attributes:\n");
    printf("sharedSizeBytes: %zu\n", attr.sharedSizeBytes);
    printf("constSizeBytes: %zu\n", attr.constSizeBytes);
    printf("localSizeBytes: %zu\n", attr.localSizeBytes);
    printf("maxThreadsPerBlock: %d\n", attr.maxThreadsPerBlock);
    printf("numRegs: %d\n", attr.numRegs);
    printf("ptxVersion: %d\n", attr.ptxVersion);
    printf("binaryVersion: %d\n", attr.binaryVersion);
    printf("cacheModeCA: %d\n", attr.cacheModeCA);
    printf("maxDynamicSharedSizeBytes: %d\n", attr.maxDynamicSharedSizeBytes);
    printf("preferredShmemCarveout: %d\n", attr.preferredShmemCarveout);
    printf("clusterDimMustBeSet: %d\n", attr.clusterDimMustBeSet);
    printf("requiredClusterWidth: %d\n", attr.requiredClusterWidth);
    printf("requiredClusterHeight: %d\n", attr.requiredClusterHeight);
    printf("requiredClusterDepth: %d\n", attr.requiredClusterDepth);
    printf("clusterSchedulingPolicyPreference: %d\n", attr.clusterSchedulingPolicyPreference);
    printf("nonPortableClusterSizeAllowed: %d\n", attr.nonPortableClusterSizeAllowed);


    long long grid_size = num_threads >> 5;
    long long block_size = 32;

    add<<<grid_size,block_size>>>(num_iterations, d_a, d_b, d_c);

    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(h_x, d_a, mem_bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_y, d_b, mem_bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_z, d_c, mem_bytes, cudaMemcpyDeviceToHost));
    
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    long res = 0;
    for (long long i = 0; i < mem_bytes; i++) {
	    res +=((char *)h_x)[i] + ((char *)h_y)[i] + ((char *)h_z)[i];
    }
    int time_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000;
    printf("Time: %d ms\n", time_ms);
    printf("Result: %ld\n\n", res);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);

    free(h_x);
    free(h_y);
    free(h_z);


    return 0;
}
