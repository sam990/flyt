#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define HANDLE_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


#define CHUNK_SIZE (1024llu * 1024llu * 4llu)
#define TOTAL_MEM_ALLOC_MB 512llu
#define TOTAL_MEM_ALLOC ((TOTAL_MEM_ALLOC_MB) * 1024llu * 1024llu)
#define NUM_CHUNKS ((TOTAL_MEM_ALLOC) / CHUNK_SIZE)

int main() {
    
    void** chunk_list = (void **)malloc(sizeof(void*) * NUM_CHUNKS);

    for (size_t i = 0; i < NUM_CHUNKS; i++) {
        HANDLE_ERROR(cudaMalloc(&chunk_list[i], CHUNK_SIZE));
    }

    for (size_t i = 0; i < NUM_CHUNKS; i++) {
        HANDLE_ERROR(cudaFree(chunk_list[i]));
    }

    free(chunk_list);

    return 0;
}
