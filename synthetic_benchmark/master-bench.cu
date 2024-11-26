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

#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
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

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threads(num_threads_per_block, num_threads_per_block);
    dim3 blocks(num_blocks, num_blocks);

    // Start benchmark
    auto start = high_resolution_clock::now();

    matrixMultiplyKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    auto end = high_resolution_clock::now();
    duration<float> duration_ms = end - start;

    // Print result and benchmark duration
    cout << "Benchmark executed in: " << duration_ms.count() * 1000 << " ms" << endl;

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void parseArguments(int argc, char *argv[], map<string, string> &params) {
    for (int i = 1; i < argc; i++) {
        stringstream ss;
        ss << argv[i + 1];
        params[argv[i]] = ss.str();
        i++; // Skip next arg
    }
}

int main(int argc, char *argv[]) {
    map<string, string> params;

    // Parse command line arguments
    parseArguments(argc, argv, params);

    // Extract parameters
    int N = stoi(params["--size"]);
    int num_threads_per_block = stoi(params["--threads_per_block"]);
    int num_blocks = stoi(params["--blocks"]);
    int task_duration_ms = stoi(params["--duration"]);

    // Run the matrix multiplication benchmark
    runMatrixMultiply(N, num_threads_per_block, num_blocks, task_duration_ms);

    return 0;
}
