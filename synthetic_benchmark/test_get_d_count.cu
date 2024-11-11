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

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        cudaStatus = cudaGetDeviceCount(&deviceCount);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaGetDeviceCount failed baby: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the time taken
    std::cout << "Time taken for " << numIterations << " calls to cudaGetDeviceCount: " 
              << elapsed.count() << " seconds" << std::endl;

    std::cout << "Total CUDA devices: " << deviceCount << std::endl;

    return 0;
}
