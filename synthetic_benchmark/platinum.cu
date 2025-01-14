#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include "logger.h"

#define GRID_SIZE 64
#define BLOCK_SIZE 192
#define ITERATIONS 10000
#define NUM_THREADS 1
#define NUM_LAUNCHES 1
#define LAUNCH_TIME_THRESHOLD 10.0f // Time threshold in milliseconds

#define CHECK_CUDA(call) {                                                   \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        std::cerr << "CUDA Runtime API error: " << cudaGetErrorString(err)   \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}


// Define the kernel with arithmetic operations
__global__ void workload_kernel(long *data, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x + threadIdx.x + 32;

    for (long k = 0; k < ITERATIONS; k++) {
        for (long j = 0; j < ITERATIONS; j++) {
            i += b;
            i -= b;
            i *= b;
        }
    }

    data[idx] += 2 * i;
    // d_a[idx] += i + d_b[idx];
}

// Function to launch kernels in a specific CUDA stream
void kernelLaunchFunction(int numLaunches, float launchTimeThreshold) {
    cudaEvent_t startEvent, stopEvent;
    cudaError_t err;

    //FlytLogger *logger = new FlytLogger(pthread_self());

    // Allocate memory for device arrays
    long *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMalloc(&d_b, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMalloc(&d_c, GRID_SIZE * BLOCK_SIZE * sizeof(long)));

    // Initialize device arrays
    CHECK_CUDA(cudaMemset(d_a, 0, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMemset(d_b, 1, GRID_SIZE * BLOCK_SIZE * sizeof(long)));
    CHECK_CUDA(cudaMemset(d_c, 0, GRID_SIZE * BLOCK_SIZE * sizeof(long)));

    bool keepLaunching = true;
    int launchCount = 0;
    // Convert the current thread ID to a string
    std::stringstream ss;
    ss << std::this_thread::get_id();
    std::string filename = ss.str() + ".txt";
    
    // Open the file in write mode
    std::ofstream file(filename, std::ios::out | std::ios::app);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
 
	cudaStream_t stream;

	CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaEventCreate(&startEvent));
        CHECK_CUDA(cudaEventCreate(&stopEvent));

    //file << "Thread id "<< pthread_self() << " flag " << flag << std::endl;
    while (keepLaunching && launchCount < numLaunches) {
        // Record start time
        //CHECK_CUDA(cudaEventRecord(startEvent, stream));
	auto start = std::chrono::high_resolution_clock::now();

        // Launch the kernel
        workload_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_a, launchCount);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA add Error: " << cudaGetErrorString(err) << std::endl;
	    //cudaStreamDestroy(stream);
	    //cudaStreamCreate(&stream);
            //return ;
        }
        cudaStreamSynchronize(stream); // Ensure the kernel launch is completed
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA sync Error: " << cudaGetErrorString(err) << std::endl;
	    //cudaStreamDestroy(stream);
	    //cudaStreamCreate(&stream);
            //return ;
        }

	auto end = std::chrono::high_resolution_clock::now();

        // Record stop time
        //CHECK_CUDA(cudaEventRecord(stopEvent, stream));

        // Calculate elapsed time
        float elapsedTime;
        //cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

	std::chrono::duration<double> diff = end - start;

	elapsedTime = (float) (1000 * diff.count());

	//logger->writeLatencyLog(elapsedTime, 810, 780);
	std::cout << "elapsedTime " << elapsedTime <<std::endl;
	std::cout << "systemelapse " << diff.count() <<std::endl;

        // Print the launch time
        //file << "Kernel launch time: " << elapsedTime << " ms" << " launch count " << launchCount << " thread-id " << pthread_self() << std::endl;

        ++launchCount;

    }
        //CHECK_CUDA(cudaEventDestroy(startEvent));
        //CHECK_CUDA(cudaEventDestroy(stopEvent));
	CHECK_CUDA(cudaStreamDestroy(stream));

    //CHECK_CUDA(cudaStreamSynchronize(stream)); // Ensure the kernel launch is completed

    // Verify results
    /*
        long *h_c = new long[GRID_SIZE * BLOCK_SIZE];
        CHECK_CUDA(cudaMemcpy(h_c, d_c, GRID_SIZE * BLOCK_SIZE * sizeof(long), cudaMemcpyDeviceToHost));

        long sum = 0;
        for (int i = 0; i < GRID_SIZE * BLOCK_SIZE; ++i) {
            sum += h_c[i];
        }

        std::cout << "sum " << sum << std::endl;
	*/
    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    std::vector<cudaStream_t> streams(NUM_THREADS);
    std::vector<std::thread> threads;

    // Create CUDA streams
    /*
    for (int i = 0; i < NUM_THREADS; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

    }
    */

    // Launch kernel functions in multiple threads with separate CUDA streams
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(kernelLaunchFunction, NUM_LAUNCHES, LAUNCH_TIME_THRESHOLD);
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Destroy CUDA streams
    /*
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }
    */

    return 0;
}

