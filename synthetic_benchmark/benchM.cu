
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#define HANDLE_ERROR(call)                                                     \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            printf("CUDA error: %s\n", cudaGetErrorString(err));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


volatile int exit_flag = 0;


void sigint_handler(int signo) { exit_flag = 1; }


__global__ void add(long iterations, long *d_a, long *d_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.x + threadIdx.x + 32;


    for (long k = 0; k < iterations; k++) {

        for (long j = 0; j < iterations; j++) {
            i += b;
            i -= b;
            i *= b;
        }
    }

    d_a[idx] += i + d_b[idx];
}


void launch_kernel(long *d_a, long *d_b, long list_size, long iterations,
                   long *h_b)
{

    int grid_size = list_size >> 5;
    int block_size = 32;

    timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    add<<<grid_size, block_size>>>(iterations, d_a, d_b);
    HANDLE_ERROR(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    HANDLE_ERROR(
        cudaMemcpy(h_b, d_a, list_size * sizeof(long), cudaMemcpyDeviceToHost));

    long h_a = 0;
    for (long i = 0; i < list_size; i++) {
        h_a += h_b[i];
    }

    int time_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000 +
                  (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000;
    printf("Time: %d ms\n", time_ms);
    printf("Result: %ld\n\n", h_a);
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s <mem_mb>\n", argv[0]);
        return 1;
    }

    printf("Pid: %d\n", getpid());

    long mem_size = strtol(argv[1], NULL, 10);
    long mem_size_bytes = mem_size << 20;

    signal(SIGINT, sigint_handler);

    // cudaError_t err;
    long list_size = 500;
    long long list_size_bytes = list_size * sizeof(long);

    long iterations = 5000;

    long *h_b = (long *)malloc(list_size_bytes);

    for (long i = 0; i < list_size; i++) {
        h_b[i] = i * 2;
    }


    long *d_a;
    long *d_b;
    long *d_total_mem;

    HANDLE_ERROR(cudaMalloc((void **)&d_a, list_size_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&d_b, list_size_bytes));
    HANDLE_ERROR(cudaMalloc((void **)&d_total_mem, mem_size_bytes));

    HANDLE_ERROR(cudaMemcpy(d_b, h_b, list_size_bytes, cudaMemcpyHostToDevice));

    while (!exit_flag) {
        HANDLE_ERROR(cudaMemset(d_a, 0, list_size_bytes));
        launch_kernel(d_a, d_b, list_size, iterations, h_b);
        sleep(1);
    }

    HANDLE_ERROR(cudaFree(d_a));
    HANDLE_ERROR(cudaFree(d_b));
    HANDLE_ERROR(cudaFree(d_total_mem));

    free(h_b);

    printf("Exiting...\n");

    return 0;
}

//  ./a.out 8000 5000
// nvcc cudaCtxLimit.cu -O0 -arch sm_86 -o cudaCtxLimit -lcuda
