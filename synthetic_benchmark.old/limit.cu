#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cstdlib>  // For atoi

#define CHECK_CUDA(call) {                                                       \
    CUresult res = (call);                                                       \
    if (res != CUDA_SUCCESS) {                                                   \
        const char *err_str;                                                     \
        cuGetErrorName(res, &err_str);                                           \
        std::cerr << "CUDA Error: " << err_str << " at " << __LINE__ << std::endl;\
        exit(1);                                                                 \
    }                                                                            \
}

// Kernel to perform some computations
__global__ void workload_kernel(int *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2;
    }
}

// Helper function to measure execution time
double execute_workload(int *data, int size, int blocks, int threads) {
    int *d_data;

    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch the kernel with reserved cores
    workload_kernel<<<blocks, threads>>>(d_data, size);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    
    cudaMemcpy(data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    return diff.count();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_SM> <threads>" << std::endl;
        return 1;
    }

    unsigned int smCount = atoi(argv[1]);  // Get number of SMs from input

    // Initialize CUDA driver API
    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    // Create context affinity parameter
    CUexecAffinityParam affinity_param = {};
    affinity_param.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;  // Set affinity type to SM count
    affinity_param.param.smCount.val = smCount;  // Set the number of SMs using the union field

    // Create the CUDA context with the specified SM count
    CUcontext ctx;
    CHECK_CUDA(cuCtxCreate_v3(&ctx, &affinity_param, 1, 0, device));

    // Set the context as the current context
    CHECK_CUDA(cuCtxSetCurrent(ctx));

    // Initialize CUDA driver API
    CHECK_CUDA(cuInit(0));

    int num_cores = 32;  // Set this to the number of SMs you want to reserve
    int num_threads_per_block = atoi(argv[2]);

    // Adjust workload to fit within the reserved SMs
    int workload_size_fit = num_cores * num_threads_per_block * 2; // Example size fitting within SMs
    int workload_size_large = workload_size_fit * 2;               // Example size larger than SMs

    int *data = new int[workload_size_large];

    // Initialize data
    for (int i = 0; i < workload_size_large; i++) {
        data[i] = i;
    }

    // Measure execution time for the workload that fits the reserved cores
    std::cout << "Running workload that fits within reserved SMs...\n";
    double time_fit = execute_workload(data, workload_size_fit, num_cores, num_threads_per_block);
    std::cout << "Execution time (fit): " << time_fit << " seconds\n";

    std::cout << "Running workload that fits within reserved SMs...\n";
    time_fit = execute_workload(data, workload_size_fit, num_cores, num_threads_per_block);
    std::cout << "Execution time (fit): " << time_fit << " seconds\n";

    std::cout << "Running workload that fits within reserved SMs...\n";
    time_fit = execute_workload(data, workload_size_fit, num_cores, num_threads_per_block);
    std::cout << "Execution time (fit): " << time_fit << " seconds\n";

    // Measure execution time for the larger workload
    std::cout << "Running larger workload that exceeds reserved SMs...\n";
    double time_large = execute_workload(data, workload_size_large, num_cores, num_threads_per_block);
    std::cout << "Execution time (large): " << time_large << " seconds\n";

    // Measure execution time for the larger workload
/*
    std::cout << "Running larger workload that exceeds reserved SMs...\n";
    time_large = execute_workload(data, workload_size_large, num_cores, num_threads_per_block);
    std::cout << "Execution time (large): " << time_large << " seconds\n";
*/

    CHECK_CUDA(cuCtxGetExecAffinity(&affinity_param, CU_EXEC_AFFINITY_TYPE_SM_COUNT));

    std::cout << "sm affinity is "<< affinity_param.param.smCount.val << "\n";

    delete[] data;
    // Clean up and destroy the context
    CHECK_CUDA(cuCtxDestroy(ctx));
    return 0;
}

