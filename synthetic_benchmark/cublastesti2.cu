#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                      \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            printf("CUDA error %d: %s\n", err, cudaGetErrorString(err)); \
            return -1;                                       \
        }                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                   \
    do {                                                     \
        cublasStatus_t status = call;                        \
        if (status != CUBLAS_STATUS_SUCCESS) {               \
            printf("cuBLAS error %d\n", status);             \
            return -1;                                       \
        }                                                    \
    } while (0)

int main() {
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Matrix dimensions
    const int m = 2, n = 3, k = 4;

    // Host data
    float A[m * k] = {1, 2, 3, 4, 5, 6, 7, 8};
    float B[k * n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float C[m * n] = {0}; // Result matrix
    float bias[n] = {1, 2, 3}; // Bias vector to add to each row of the result

    // Device memory
    float *d_A, *d_B, *d_C, *d_bias;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(A)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(B)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(C)));
    CHECK_CUDA(cudaMalloc((void**)&d_bias, sizeof(bias)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(B), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, sizeof(bias), cudaMemcpyHostToDevice));

    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set the bias pointer attribute
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc,
                                                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &d_bias,
                                                sizeof(d_bias)));

    // Create matrix layouts
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, k)); // Leading dimension = k
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, n)); // Leading dimension = n
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, n)); // Leading dimension = n

    // Allocate workspace
    void *workspace = nullptr;
    size_t workspaceSize = 4 * 1024 * 1024; // 4 MB workspace (adjust as needed)
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Initialize heuristic structure and get algorithm
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspaceSize,
                                                      sizeof(workspaceSize)));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle,
                                                operationDesc,
                                                Adesc,
                                                Bdesc,
                                                Cdesc,
                                                Cdesc,
                                                preference,
                                                1, // Request only one algorithm
                                                &heuristicResult,
                                                &returnedResults));

    if (returnedResults == 0) {
        printf("No suitable algorithm found.\n");
        return -1;
    }
    algo = heuristicResult.algo;

    // Execute the matrix multiplication with bias
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasLtMatmul(handle,
                                operationDesc,
                                &alpha,
                                d_A, Adesc,
                                d_B, Bdesc,
                                &beta,
                                d_C, Cdesc,
                                d_C, Cdesc,
                                &algo, workspace, workspaceSize, 0));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(C), cudaMemcpyDeviceToHost));

    // Print the result
    printf("Result matrix:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    // Clean up
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    //CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(workspace));

    return 0;
}

