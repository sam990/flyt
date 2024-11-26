#ifndef _CD_COMMON_H_
#define _CD_COMMON_H_

#include <rpc/rpc.h>
#include <pthread.h>
#include "list.h"
#include <time.h>
#include <stdio.h>

#define CD_SOCKET_PATH "/tmp/cricketd_sock"
#ifndef LOG_LEVEL
    #define LOG_LEVEL LOG_DEBUG
#endif //LOG_LEVEL

typedef struct kernel_info {
    char *name;
    size_t param_size;
    size_t param_num;
    uint16_t *param_offsets;
    uint16_t *param_sizes;
    void *host_fun;
} kernel_info_t;

enum socktype_t {UNIX, TCP, UDP} socktype;
#define INIT_SOCKTYPE enum socktype_t socktype = TCP;

int connection_is_local;
int shm_enabled;
//#define INIT_SOCKTYPE enum socktype_t socktype = UNIX;
#define WITH_API_CNT
//#define WITH_IB


CLIENT *clnt;
list kernel_infos;

// semaphore
pthread_rwlock_t access_sem;

struct rpc_times {
    double rpc_init_1;
    double rpc_elf_load_1;
    double rpc_register_function_1;
    double rpc_register_var_1;
    double cuda_get_device_count_1;
    double cuda_get_device_properties_1;
    double cuda_get_device_1;
    double rpc_cudeviceprimaryctxgetstate_1;
    double cuda_device_get_stream_priority_range_1;
    double cuda_stream_is_capturing_1;
    double cuda_get_last_error_1;
    double cuda_malloc_1;
    double cudaMemcpyH2D;
    double cudaMemcpyD2H;
    double cudaMemcpyD2D;
    double cuda_stream_synchronize_1;
    double cuda_device_synchronize_1;
    double cuda_launch_kernel_1;
    double rpc_cuctxgetcurrent_1;
    double cublasCreate_v2;
    double rpc_cublassetstream_1;
    double rpc_cublassetworkspace_1;
    double rpc_cublassetmathmode_1;
    double rpc_cublassgemm_1; 
    double rpc_cublasltmatmuldesccreate_1;
    double rpc_cublasltmatmuldescsetattribute_1;
    double rpc_cublasltmatrixlayoutcreate_1;
    double rpc_cublasltmatmulpreferencecreate_1;
    double rpc_cublasltmatmulalgogetheuristic_1;
    double rpc_cublasltmatmul_1;
    double rpc_cublasltmatrixlayoutdestroy_1;
    double cuda_device_get_attribute_1;
    double cuda_func_get_attributes_1;
    double cuda_occupancy_max_active_bpm_with_flags_1;
    double rpc_cublasltmatmulpreferencedestroy_1;
    double rpc_cublasltmatmuldescdestroy_1;  
    double rpc_cudeviceget_1;  
    double cuda_choose_device_1;
};

struct rpc_cnt {
    double rpc_init_1;
    double rpc_elf_load_1;
    double rpc_register_function_1;
    double rpc_register_var_1;
    double cuda_get_device_count_1;
    double cuda_get_device_properties_1;
    double cuda_get_device_1;
    double rpc_cudeviceprimaryctxgetstate_1;
    double cuda_device_get_stream_priority_range_1;
    double cuda_stream_is_capturing_1;
    double cuda_get_last_error_1;
    double cuda_malloc_1;
    double cudaMemcpyH2D;
    double cudaMemcpyD2H;
    double cudaMemcpyD2D;
    double cuda_stream_synchronize_1;
    double cuda_device_synchronize_1;
    double cuda_launch_kernel_1;
    double rpc_cuctxgetcurrent_1;
    double cublasCreate_v2;
    double rpc_cublassetstream_1;
    double rpc_cublassetworkspace_1;
    double rpc_cublassetmathmode_1;
    double rpc_cublassgemm_1; 
    double rpc_cublasltmatmuldesccreate_1;
    double rpc_cublasltmatmuldescsetattribute_1;
    double rpc_cublasltmatrixlayoutcreate_1;
    double rpc_cublasltmatmulpreferencecreate_1;
    double rpc_cublasltmatmulalgogetheuristic_1;
    double rpc_cublasltmatmul_1;
    double rpc_cublasltmatrixlayoutdestroy_1;
    double cuda_device_get_attribute_1;
    double cuda_func_get_attributes_1;
    double cuda_occupancy_max_active_bpm_with_flags_1;
    double rpc_cublasltmatmulpreferencedestroy_1;
    double rpc_cublasltmatmuldescdestroy_1;
    double rpc_cudeviceget_1;
    double cuda_choose_device_1;
};

// defined in constructor
extern struct rpc_times *times__flyt; 
extern struct rpc_cnt *counts__flyt;

long get_time_ms(); 
double calc_total_rpc_time();

struct rpc_times *init_rpc_times(); // from constructor.
struct rpc_cnt *init_rpc_counts();

void report_rpc_stats(); // from destructor.

typedef struct {
    double start_time;
    double end_time;
} Timer;

#define FIRST_ARG(...) _FIRST_ARG(__VA_ARGS__,)
#define _FIRST_ARG(first, ...) first

#define FUNC_BEGIN(...) \
    do { \
        if (sizeof((Timer[]){__VA_ARGS__}) / sizeof(Timer) == 0) { \
            pthread_rwlock_rdlock(&access_sem); \
        } else { \
            __VA_OPT__(FIRST_ARG(__VA_ARGS__).start_time = get_time_ms(); \
            pthread_rwlock_rdlock(&access_sem);) \
        } \
    } while(0)

#define FUNC_END(...) \
    pthread_rwlock_unlock(&access_sem)

#define TIMER_ADD_INCREMENT(timer, callname) \
    do { \
        (timer).end_time = get_time_ms(); \
        long incr = (timer).end_time - (timer).start_time; \
        times__flyt->callname += incr; \
        counts__flyt->callname += 1; \
    } while(0);

#endif //_CD_COMMON_H_

