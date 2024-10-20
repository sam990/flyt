#include "cpu-common.h"
#include <stdlib.h>

long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000) + (ts.tv_nsec / 1000000);
}

struct rpc_times *init_rpc_times() {
    struct rpc_times *_times = malloc(sizeof(struct rpc_times));
    if (_times!= NULL) {
        memset(_times, 0, sizeof(struct rpc_times));
    }

    return _times;   
}

struct rpc_cnt *init_rpc_counts() {
    struct rpc_cnt *_calls = malloc(sizeof(struct rpc_cnt));
    if (_calls != NULL) {
        memset(_calls, 0, sizeof(struct rpc_cnt));
    }

    return _calls;    
}

void report_rpc_stats() {
    const char *function_names[] = {
        "rpc_elf_load",
        // "rpc_register_function",
        // "rpc_register_var",
        // "cudaGetDeviceCount",
        // "cudaGetDeviceProperties",
        // more funcs
    };

    long function_times[] = {
        times->t_rpc_elf_load_1,
        // times->t_rpc_register_function,
        // times->t_rpc_register_var,
        // times->t_cudaGetDeviceCount,
        // times->t_cudaGetDeviceProperties,
        // other time pointers
    };

    long function_counts[] = {
        calls->cnt_rpc_elf_load_1,
        // calls->cnt_rpc_register_function,
        // calls->cnt_rpc_register_var,
        // calls->cnt_cudaGetDeviceCount,
        // calls->cnt_cudaGetDeviceProperties,
    };

    size_t num_functions = sizeof(function_times) / sizeof(function_times[0]);

    printf("RPC Call Timing Report:\n");
    for (size_t i = 0; i < num_functions; ++i) {
        printf("[%s]: %ld calls, %ld ms\n", function_names[i], function_counts[i], function_times[i]);
    }

}