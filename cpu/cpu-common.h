#ifndef _CD_COMMON_H_
#define _CD_COMMON_H_

#include <rpc/rpc.h>
#include <pthread.h>
#include "list.h"

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

static long start_time;
static long end_time;

struct rpc_times {
    long t_rpc_elf_load_1;
    // long t_rpc_register_function;
    // long t_rpc_register_var;
    // long t_cudaGetDeviceCount;
    // long t_cudaGetDeviceProperties;
    // add more
};

struct rpc_cnt {
    long cnt_rpc_elf_load_1;
    // long cnt_rpc_register_function;
    // long cnt_rpc_register_var;
    // long cnt_cudaGetDeviceCount;
    // long cnt_cudaGetDeviceProperties;
    // other cnts
};

// defined in constructor
extern struct rpc_times *times; 
extern struct rpc_cnt *calls;

long get_time_ms(); 

struct rpc_times *init_rpc_times(); // from constructor.
struct rpc_cnt *init_rpc_counts();

void report_rpc_stats(); // from destructor.

#define FUNC_BEGIN \
    start_time = get_time_ms(); \
    pthread_rwlock_rdlock(&access_sem);

#define FUNC_END \
    pthread_rwlock_unlock(&access_sem);

#define TIMER_ADD_INCREMENT(callname) \
    do { \
        end_time = get_time_ms(); \
        times->t_##callname += (end_time - start_time); \
        calls->cnt_##callname += 1; \
    } while(0)

#endif //_CD_COMMON_H_

