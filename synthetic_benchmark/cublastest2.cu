#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS API failed with status: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    cublasHandle_t handle;
    checkCublasStatus(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    int num_iterations = 20;
    int N = 1024;  // Adjust size for testing
    cudaMalloc((void**)&d_A, sizeof(float) * N * N);
    cudaMalloc((void**)&d_B, sizeof(float) * N * N);
    cudaMalloc((void**)&d_C, sizeof(float) * N * N);

    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < num_iterations; ++i) {
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            N, N, N, &alpha,
                                            d_A, N, d_B, N, &beta, d_C, N);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasSgemm failed at iteration " << i << " with status: " << status << std::endl;
            break;
        }
        cudaDeviceSynchronize();
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}

