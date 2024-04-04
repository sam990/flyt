#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <gpu-device-id>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int device_id = atoi(argv[1]);

    cudaDeviceProp dev_prop;
    
    cudaError_t res = cudaGetDeviceProperties(&dev_prop, device_id);


    if (res != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(res));
        exit(EXIT_FAILURE);
    }

    printf("%u", dev_prop.multiProcessorCount);

    return 0;

}