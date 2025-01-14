//#include <iostream>
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
    cublasStatus_t status = cublasCreate(&handle);
    checkCublasStatus(status);

    // Example matrices and cuBLAS operation
    // Ensure your matrices, pointers and operations are valid here
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * 4);
    cudaMalloc((void**)&d_B, sizeof(float) * 4);
    cudaMalloc((void**)&d_C, sizeof(float) * 4);

    // Example cuBLAS function: cublasSgemm (Matrix multiplication)
    float alpha = 1.0f, beta = 0.0f;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, d_A, 2, d_B, 2, &beta, d_C, 2);
    checkCublasStatus(status);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

