/**
 * FILE: master-bench.cu
 * ---------------------
 * HPC: High #calls, high #threads (1024 *1024), large, frequent memory transfers,
 * compute intensive kernels (matrix multiply)
 * 
 * Graphics: High #calls, small #threads (8 * 8), 
 *
 * Video Processing: Low #calls, #threads depend on resolution,
 * memcpy per frame, fixed patterns
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

__global__ void matrixMultiplyKernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void runMatrixMultiply(int N, int num_threads_per_block, int num_blocks, int task_duration_ms) {
    float *A, *B, *C;
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize matrices with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_A, size);
    HANDLE_CUDA_ERROR(err);

    err = cudaMalloc(&d_B, size);
    HANDLE_CUDA_ERROR(err);

    err = cudaMalloc(&d_C, size);
    HANDLE_CUDA_ERROR(err);

    // Copy data from host to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    HANDLE_CUDA_ERROR(err);

    // Set up grid and block dimensions
    dim3 threads(num_threads_per_block, num_threads_per_block);
    dim3 blocks(num_blocks, num_blocks);

    // Start benchmark using gettimeofday to track duration
    struct timeval start, end;
    gettimeofday(&start, NULL);

    long long elapsed_ms = 0; // Elapsed time in milliseconds
    long long task_duration_us = task_duration_ms * 1000; // Task duration in microseconds

    while (elapsed_ms < task_duration_ms) {
        // Run the matrix multiplication kernel
        matrixMultiplyKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the result back to the host
        err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
        HANDLE_CUDA_ERROR(err);

        // Get the elapsed time and check if the task duration has passed
        gettimeofday(&end, NULL);
        elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000;
    }

    // Print the total duration
    printf("Benchmark executed for: %lld milliseconds\n", elapsed_ms);

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void printUsage() {
    printf("Usage: ./master-bench --size <matrix_size> --threads_per_block <threads_per_block> "
           "--duration <duration_ms> [--blocks <blocks>]\n");
    printf("    --size <matrix_size>      : The size of the matrix (e.g., 1024 for 1024x1024 matrix)\n");
    printf("    --threads_per_block <threads_per_block> : Number of threads per block (e.g., 16)\n");
    printf("    --duration <duration_ms>  : Duration of the benchmark in milliseconds (e.g., 2000)\n");
    printf("    --blocks <blocks>         : Number of blocks in the grid (optional, default: 16)\n");
}

int main(int argc, char *argv[]) {
    int opt;
    int N = 0, num_threads_per_block = 0, num_blocks = 16, task_duration_ms = 0; // Default num_blocks to 16

    // Parse command line arguments using getopt
    while ((opt = getopt(argc, argv, "s:t:d:b:")) != -1) {
        switch (opt) {
            case 's':
                N = atoi(optarg); // Size of the matrix
                break;
            case 't':
                num_threads_per_block = atoi(optarg); // Threads per block
                break;
            case 'd':
                task_duration_ms = atoi(optarg); // Task duration in ms
                break;
            case 'b':
                num_blocks = atoi(optarg); // Number of blocks in grid
                break;
            default:
                printUsage();
                exit(EXIT_FAILURE);
        }
    }

    // Ensure all required arguments are provided
    if (N == 0 || num_threads_per_block == 0 || task_duration_ms == 0) {
        printUsage();
        exit(EXIT_FAILURE);
    }

    // Print the parsed arguments
    printf("Matrix size: %d x %d\n", N, N);
    printf("Threads per block: %d\n", num_threads_per_block);
    printf("Blocks: %d\n", num_blocks);
    printf("Task duration: %d ms\n", task_duration_ms);

    // Run the matrix multiplication benchmark
    runMatrixMultiply(N, num_threads_per_block, num_blocks, task_duration_ms);

    return 0;
}
