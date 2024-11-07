/**
  * FILE: cpu-common-stats.c
  * ------------------------
  * To track more calls, add to function_names, update count, 
  * wrap the actual calll with FUNC_BEGIN(), FUNC_END(), TIMER_ADD_INCREMENT()
 */

#include "cpu-common.h"
#include <stdlib.h>

long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    //return (ts.tv_sec * 1000) + (ts.tv_nsec / 1000000);
    return (ts.tv_sec * 1000000000) + (ts.tv_nsec);
}

struct rpc_times * init_rpc_times() {
    struct rpc_times *_times = malloc(sizeof(struct rpc_times)); // Use double pointer
    if (_times != NULL) {
        memset(_times, 0, sizeof(struct rpc_times));
    } else {
        printf("times malloc failed.\n");
        exit(-1);
    }
    return _times;
}

struct rpc_cnt * init_rpc_counts() {
    struct rpc_cnt * _counts = malloc(sizeof(struct rpc_cnt)); // Use double pointer
    if (_counts != NULL) {
        memset(_counts, 0, sizeof(struct rpc_cnt));
    } else {
        printf("count malloc failed.\n");
        exit(-1);
    }
    return _counts;
}

// in ms
double calc_total_rpc_time() {
    double total_time = 0.0;
    total_time += times__flyt->rpc_elf_load_1 / 1000000.0;
    total_time += times__flyt->rpc_init_1 / 1000000.0;
    total_time += times__flyt->rpc_register_function_1 / 1000000.0;
    total_time += times__flyt->rpc_register_var_1 / 1000000.0;
    total_time += times__flyt->cuda_get_device_count_1 / 1000000.0;
    total_time += times__flyt->cuda_get_device_properties_1 / 1000000.0;
    total_time += times__flyt->cuda_get_device_1 / 1000000.0;
    total_time += times__flyt->rpc_cudeviceprimaryctxgetstate_1 / 1000000.0;
    total_time += times__flyt->cuda_device_get_stream_priority_range_1 / 1000000.0;
    total_time += times__flyt->cuda_stream_is_capturing_1 / 1000000.0;
    total_time += times__flyt->cuda_get_last_error_1 / 1000000.0;
    total_time += times__flyt->cuda_malloc_1 / 1000000.0;
    total_time += times__flyt->cudaMemcpyH2D / 1000000.0;
    total_time += times__flyt->cudaMemcpyD2H / 1000000.0;
    total_time += times__flyt->cudaMemcpyD2D / 1000000.0;
    total_time += times__flyt->cuda_stream_synchronize_1 / 1000000.0;
    //total_time += times__flyt->cuda_device_synchronize_1 / 1000000.0;
    total_time += times__flyt->cuda_launch_kernel_1 / 1000000.0;
    total_time += times__flyt->rpc_cuctxgetcurrent_1 / 1000000.0;
    total_time += times__flyt->cublasCreate_v2 / 1000000.0;
    total_time += times__flyt->rpc_cublassetstream_1 / 1000000.0;
    total_time += times__flyt->rpc_cublassetworkspace_1 / 1000000.0;
    total_time += times__flyt->rpc_cublassetmathmode_1 / 1000000.0;
    total_time += times__flyt->rpc_cublassgemm_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatmuldesccreate_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatmuldescsetattribute_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatrixlayoutcreate_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatmulpreferencecreate_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatmulalgogetheuristic_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatmul_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatrixlayoutdestroy_1 / 1000000.0;
    total_time += times__flyt->rpc_cublasltmatmuldescdestroy_1 / 1000000.0;
    total_time += times__flyt->rpc_cudeviceget_1 / 1000000.0;
    return total_time;
}

// Make this better later
#define NUM_CUDA_API_FUNC_NAME_COUNT 36
void report_rpc_stats() {
    printf("RPC Call Timing Report:\n");
    printf("-----------------------------------------\n");

    printf("%-40s | %10s | %10s\n", "Function Name", "Calls", "Time (ms)");
    printf("-----------------------------------------\n");

    printf("%-40s | %10ld | %10.3f\n", "rpc_elf_load_1", (long int)counts__flyt->rpc_elf_load_1, times__flyt->rpc_elf_load_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_init_1", (long int)counts__flyt->rpc_init_1, times__flyt->rpc_init_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_register_function_1", (long int)counts__flyt->rpc_register_function_1, times__flyt->rpc_register_function_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_register_var_1", (long int)counts__flyt->rpc_register_var_1, times__flyt->rpc_register_var_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_get_device_count_1", (long int)counts__flyt->cuda_get_device_count_1, times__flyt->cuda_get_device_count_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_get_device_properties_1", (long int)counts__flyt->cuda_get_device_properties_1, times__flyt->cuda_get_device_properties_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_get_device_1", (long int)counts__flyt->cuda_get_device_1, times__flyt->cuda_get_device_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cudeviceprimaryctxgetstate_1", (long int)counts__flyt->rpc_cudeviceprimaryctxgetstate_1, times__flyt->rpc_cudeviceprimaryctxgetstate_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_device_get_stream_priority_range_1", (long int)counts__flyt->cuda_device_get_stream_priority_range_1, times__flyt->cuda_device_get_stream_priority_range_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_stream_is_capturing_1", (long int)counts__flyt->cuda_stream_is_capturing_1, times__flyt->cuda_stream_is_capturing_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_get_last_error_1", (long int)counts__flyt->cuda_get_last_error_1, times__flyt->cuda_get_last_error_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_malloc_1", (long int)counts__flyt->cuda_malloc_1, times__flyt->cuda_malloc_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cudaMemcpyH2D", (long int)counts__flyt->cudaMemcpyH2D, times__flyt->cudaMemcpyH2D / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cudaMemcpyD2H", (long int)counts__flyt->cudaMemcpyD2H, times__flyt->cudaMemcpyD2H / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cudaMemcpyD2D", (long int)counts__flyt->cudaMemcpyD2D, times__flyt->cudaMemcpyD2D / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_stream_synchronize_1", (long int)counts__flyt->cuda_stream_synchronize_1, times__flyt->cuda_stream_synchronize_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_device_synchronize_1", (long int)counts__flyt->cuda_device_synchronize_1, times__flyt->cuda_device_synchronize_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuda_launch_kernel_1", (long int)counts__flyt->cuda_launch_kernel_1, times__flyt->cuda_launch_kernel_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cuctxgetcurrent_1", (long int)counts__flyt->rpc_cuctxgetcurrent_1, times__flyt->rpc_cuctxgetcurrent_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cublasCreate_v2", (long int)counts__flyt->cublasCreate_v2, times__flyt->cublasCreate_v2 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublassetstream_1", (long int)counts__flyt->rpc_cublassetstream_1, times__flyt->rpc_cublassetstream_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublassetworkspace_1", (long int)counts__flyt->rpc_cublassetworkspace_1, times__flyt->rpc_cublassetworkspace_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublassetmathmode_1", (long int)counts__flyt->rpc_cublassetmathmode_1, times__flyt->rpc_cublassetmathmode_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublassgemm_1", (long int)counts__flyt->rpc_cublassgemm_1, times__flyt->rpc_cublassgemm_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatmuldesccreate_1", (long int)counts__flyt->rpc_cublasltmatmuldesccreate_1, times__flyt->rpc_cublasltmatmuldesccreate_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatmuldescsetattribute_1", (long int)counts__flyt->rpc_cublasltmatmuldescsetattribute_1, times__flyt->rpc_cublasltmatmuldescsetattribute_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatrixlayoutcreate_1", (long int)counts__flyt->rpc_cublasltmatrixlayoutcreate_1, times__flyt->rpc_cublasltmatrixlayoutcreate_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatmulpreferencecreate_1", (long int)counts__flyt->rpc_cublasltmatmulpreferencecreate_1, times__flyt->rpc_cublasltmatmulpreferencecreate_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatmulalgogetheuristic_1", (long int)counts__flyt->rpc_cublasltmatmulalgogetheuristic_1, times__flyt->rpc_cublasltmatmulalgogetheuristic_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatmul_1", (long int)counts__flyt->rpc_cublasltmatmul_1, times__flyt->rpc_cublasltmatmul_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatrixlayoutdestroy_1", (long int)counts__flyt->rpc_cublasltmatrixlayoutdestroy_1, times__flyt->rpc_cublasltmatrixlayoutdestroy_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "rpc_cublasltmatmuldescdestroy_1", (long int)counts__flyt->rpc_cublasltmatmuldescdestroy_1, times__flyt->rpc_cublasltmatmuldescdestroy_1 / 1000000.0);
    printf("%-40s | %10ld | %10.3f\n", "cuDeviceGet", (long int)counts__flyt->rpc_cudeviceget_1, times__flyt->rpc_cudeviceget_1 / 1000000.0);
    printf("-----------------------------------------\n");
}
