#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include "logger.h"
#include <random>
#include <thread>
#include <chrono>

#define ITERATIONS 10000

__global__ void kernelDummyLoad(int *data, int load, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= load)
	    idx = load -1;

    if (iterations < load)
	    load = iterations;

    for( int i = 0; i < load; i++) {
        data[idx] = (data[idx] + idx) % 255;
    }
    //if (idx < iterations) {
    //    for (int i = 0; i < ITERATIONS; i++) {
     //       __syncthreads();
     //   }
    //}
}

// Utility function to check for CUDA errors
void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s: %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}

int get_active_sm_count() {

	int count = 0;
	checkCudaError(cudaGetDeviceCount(&count), "cudaGetDevice");

	return count;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <number_of_sms> <seconds_per_load>\n", argv[0]);
        return -1;
    }

    int num_sms = atoi(argv[1]);        // Number of SMs to allocate
    float seconds_per_load = atof(argv[2]);  // Duration (float) in seconds for each workload cycle

    // Get device properties to determine the maxThreadsPerBlock
    cudaDeviceProp deviceProp;
    checkCudaError(cudaGetDeviceProperties(&deviceProp, 0), "Error getting device properties");

    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor;

    printf("Device has %d SMs\n", deviceProp.multiProcessorCount);
    printf("Max threads per block: %d\n", maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n", maxBlocksPerSM);

    // Ensure we don't exceed the available number of SMs
    if (num_sms > deviceProp.multiProcessorCount) {
        printf("Requested number of SMs exceeds available SMs. Reducing to max available (%d).\n", deviceProp.multiProcessorCount);
        num_sms = deviceProp.multiProcessorCount;
    }

    // Run the experiment for 5 minutes (300 seconds)
    const int experiment_durationSec = 300;
    const int printIntervalMs = 100;   // Print SM load every 0.5 seconds
    const int num_observations = (int)((seconds_per_load * 1000)/(2 * printIntervalMs));
    const int dataSize = 1024 * 1024;  // Data size to process (1 million elements)

    printf("No. of Observations used (50 of it for decision): %d\n", num_observations *2);

    int *d_data;
    checkCudaError(cudaMalloc(&d_data, dataSize * sizeof(int)), "Allocating device memory");

    //cudaContext_t context;
    //checkCudaError(cudaDeviceGetDefaultCtx(&context), "Error getting default context");

    // Create a custom stream for kernel launches
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate ");



    // Random number generation setup for load percentage (0% to 200%)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> loadDist(10.0, 100.0);  // Random load from 0% to 200%

    FlytLogger *logger = new FlytLogger(pthread_self());

    for (float t = 0; t < experiment_durationSec;) {
        // Generate a random load between 0% to 200% of the allocated SMs
//        float load_factor = ((rand() % 201) / 100.0);  // Random factor between 0.0 and 2.0
	// Generate a random load percentage from 0% to 200%
        float smLoadPercentage = loadDist(gen);

        int active_sms = (int)smLoadPercentage; //((deviceProp.multiProcessorCount * smLoadPercentage)/100);

        int blocksPerGrid = active_sms * maxBlocksPerSM;
        int threadsPerBlock = maxThreadsPerBlock;

        // Print the load that will be sustained for the next few seconds
	/*
        printf("Setting new load | Target SMs: %d | Load Factor: %.2f | Duration: %.1f seconds\n",
               active_sms, load_factor, seconds_per_load);
	       */

        // Run for full half-second intervals
        for (float cycle = 0; cycle < (seconds_per_load * 1000); ) {
	    // Get actualSM allocated.
	    int smCount = get_active_sm_count();
	    int load_level = threadsPerBlock * blocksPerGrid;
	    int load = dataSize; //(dataSize > load_level)? dataSize : load_level;
    
	    // Record start time
            auto start = std::chrono::high_resolution_clock::now();


            // Launch the kernel with the computed load
            kernelDummyLoad<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, load, ITERATIONS);
	    cudaStreamSynchronize(stream);

	    // Record end time
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> durationMs = end - start;

            // Compute the actual SMs used
            int actual_sms_used = blocksPerGrid / maxBlocksPerSM;
	    static char name[100];
	    sprintf(name,"%04d:%04d", actual_sms_used,smCount);

            // Print the current load and time information
            printf("Execution time: %.1f ms | SMs reserved : %d | Load (Threads) : %d | fn name %s\n",
                   durationMs.count(), smCount, actual_sms_used, name);

	    // 50% of observations need to be higher or greater..)
	    logger->writelog(name, (int)actual_sms_used, (int)smCount, (int)smCount, (int)num_observations);

	    // Ensure that the loop completes every 500 ms (including kernel execution)
            int remainingTimeMs = printIntervalMs - static_cast<int>(durationMs.count());
            if (remainingTimeMs > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(remainingTimeMs));
            }
	    else {
		remainingTimeMs = 0;
	    }

            cycle += remainingTimeMs + durationMs.count();  
            t += (remainingTimeMs + durationMs.count()) / 1000;  
        }

    }
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    printf("Execution of program completed\n");

    return 0;
}

