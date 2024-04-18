#include <cuda.h>
#include <cuda_runtime.h>

int get_gpu_cores(unsigned device_id) {
    struct cudaDeviceProp dev_prop;
    cudaError_t res = cudaGetDeviceProperties(&dev_prop, device_id);
    if (res != cudaSuccess) {
        return -1;
    }
    return dev_prop.multiProcessorCount;
}