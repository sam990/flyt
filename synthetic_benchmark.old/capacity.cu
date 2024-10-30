#include <iostream>
#include <cuda.h>
#include "logger.h"

// Kernel function
__global__ void test_kernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 10000; i++) {  // Just a dummy operation to keep the kernel busy
            data[idx] += 1;
        }
    }
}

// Function to measure the kernel execution time
float measure_kernel_execution(int blocks, int threads_per_block, int *d_data, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    test_kernel<<<blocks, threads_per_block>>>(d_data, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

int main() {
    const int N = 1 << 20;  // Data size
    int *h_data, *d_data;
    h_data = new int[N];
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 1536;
    int blocks_32_sms = 10 * 4;  // Launch enough blocks to use 32 SMs
    int blocks_64_sms = 64 * 8;  // Launch enough blocks to use 64 SMs

    // Measure execution time with 32 SMs
    float time_32_sms = measure_kernel_execution(blocks_32_sms, threads_per_block, d_data, N);
    std::cout << "Kernel execution time with 32 SMs: " << time_32_sms << " ms\n";

    // Measure execution time with 64 SMs
    float time_64_sms = measure_kernel_execution(blocks_64_sms, threads_per_block, d_data, N);
    std::cout << "Kernel execution time with 64 SMs: " << time_64_sms << " ms\n";

    // Cleanup
    cudaFree(d_data);
    delete[] h_data;

    return 0;
}

