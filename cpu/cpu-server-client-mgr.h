# ifndef __FLYT_CPU_SERVER_CLIENT_MGR_H
# define __FLYT_CPU_SERVER_CLIENT_MGR_H

#include "resource-mg.h"
#include "resource-map.h"

#define INIT_MEM_SLOTS 4096
#define INIT_STREAM_SLOTS 16
#define INIT_MODULE_SLOTS 8
#define INIT_FUNCTION_SLOTS 128
#define INIT_VAR_SLOTS 128

typedef struct __addr_data_pair {
    void* addr;
    void* reg_data;
} addr_data_pair_t;

typedef struct __cricket_client {
    int pid;
    resource_map* gpu_mem;
    void* default_stream;
    resource_map* custom_streams;
    resource_mg modules;
    resource_mg functions;
    resource_mg vars;
    // further can be added
} cricket_client;

typedef uint64_t cricket_client_iter;


int init_cpu_server_client_mgr();

void free_cpu_server_client_mgr();

cricket_client *create_client(int pid);

int add_new_client(int pid, int xp_fd);

int add_restored_client(cricket_client *client);

int move_restored_client(int pid, int xp_fd);

cricket_client* get_client(int xp_fd);

cricket_client* get_client_by_pid(int pid);

int remove_client(int xp_fd);

cricket_client_iter get_client_iter();

cricket_client *get_next_client(cricket_client_iter *iter);

cricket_client *get_next_restored_client(cricket_client_iter *iter);

int remove_client_by_pid(int pid);

#endif // __FLYT_CPU_SERVER_CLIENT_MGR_H