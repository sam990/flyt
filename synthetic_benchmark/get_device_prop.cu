#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib> // For std::atoi

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <numIterations>" << std::endl;
        return 1;
    }

    int numIterations = std::atoi(argv[1]); // Convert the argument to an integer
    int deviceCount;
    cudaError_t cudaStatus;

    // Get the number of CUDA devices
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    // Assume querying properties of the first device
    int deviceToQuery = 0;
    cudaDeviceProp deviceProp;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        cudaStatus = cudaGetDeviceProperties(&deviceProp, deviceToQuery);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the time taken
    std::cout << "Time taken for " << numIterations << " calls to cudaGetDeviceProperties: " 
              << elapsed.count() << " seconds" << std::endl;

    // Output some properties of the device as verification
    std::cout << "Queried device: " << deviceToQuery << std::endl;
    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;

    return 0;
}