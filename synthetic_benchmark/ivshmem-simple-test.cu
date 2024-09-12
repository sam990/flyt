#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size of data in bytes>\n", argv[0]);
        return 1;
    }

    // Parse the size of data to be copied from the command line
    size_t data_size = atol(argv[1]);

    // Allocate host memory
    char *host_data = (char *)malloc(data_size);
    char *host_data_copy = (char *)malloc(data_size); // For verifying d2h data

    if (!host_data || !host_data_copy) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Initialize host data with a pattern that fills the entire size
    for (size_t i = 0; i < data_size; ++i) {
        host_data[i] = (char)(i % 256); // Fill with a repeating pattern
    }

    // Allocate device memory
    char *device_data;
    cudaError_t err = cudaMalloc((void **)&device_data, data_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(host_data);
        free(host_data_copy);
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(device_data, host_data, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (Host to Device) failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_data);
        free(host_data);
        free(host_data_copy);
        return 1;
    }

    // Copy data back from device to host
    err = cudaMemcpy(host_data_copy, device_data, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (Device to Host) failed: %s\n", cudaGetErrorString(err));
        cudaFree(device_data);
        free(host_data);
        free(host_data_copy);
        return 1;
    }

    // Verify data
    if (memcmp(host_data, host_data_copy, data_size) == 0) {
        printf("Data verification successful: Host to Device to Host copy matches.\n");
    } else {
        printf("Data verification failed: Mismatch between original and copied data.\n");
    }

    // Cleanup
    cudaFree(device_data);
    free(host_data);
    free(host_data_copy);

    return 0;
}
