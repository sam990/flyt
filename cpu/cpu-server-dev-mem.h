#ifndef __CPU_SERVER_DEV_MEM
#define __CPU_SERVER_DEV_MEM

#include <cuda_runtime.h>

int init_server_dev_mem(int device_id);
cudaError_t dev_mem_alloc(void **dev_ptr, size_t size, int va_specified, size_t *padded_size_out);
cudaError_t dev_mem_alloc_3d(void **dev_ptr, size_t depth, size_t height, size_t width, size_t *pitch_out, size_t *padded_size_out);
cudaError_t dev_mem_alloc_pitched(void **dev_ptr, size_t width, size_t height, size_t *pitch_out, size_t *padded_size_out);
cudaError_t free_dev_mem(void *ptr, size_t padded_size);

#endif