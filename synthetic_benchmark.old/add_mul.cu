#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cuda.h>

__global__ void add(long iterations, long *d_a) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x + threadIdx.x + 32;


    for (long k = 0; k < iterations; k++) {

        for (long j = 0; j < iterations; j++) {
            i += b;
            i -= b;
            i *= b;
        }
    }
    
    *d_a += i;
}


int main (int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <list-size> <iterations>\n", argv[0]);
        return 1;
    } 

    long list_size = strtol(argv[1], NULL, 16);
    long iterations = strtol(argv[2], NULL, 10);
    
    // printf("List size: %ld\n", list_size);
    // printf("Iterations: %ld\n", iterations);

    int grid_size = list_size >> 5;
    int block_size = 32;

    long *d_a;
    cudaMalloc((void**)&d_a, sizeof(long));
    cudaMemset(d_a, 0, sizeof(long));

    timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    add<<<grid_size,block_size>>>(iterations, d_a);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    long h_a;
    cudaMemcpy(&h_a, d_a, sizeof(long), cudaMemcpyDeviceToHost);

    cudaFree(d_a);

    int time_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000;
    printf("Time: %d ms\n", time_ms);
    printf("Result: %ld\n", h_a);

    return 0;
}

//  ./a.out 8000 5000