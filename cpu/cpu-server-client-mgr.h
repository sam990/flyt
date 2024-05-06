# ifndef __FLYT_CPU_SERVER_CLIENT_MGR_H
# define __FLYT_CPU_SERVER_CLIENT_MGR_H

#include "resource-map.h"

#define INIT_MEM_SLOTS 4096
#define INIT_STREAM_SLOTS 16



typedef struct __cricket_client {
    int pid;
    resource_map* gpu_mem;
    void* default_stream;
    resource_map* custom_streams;
    // further can be added
} cricket_client;


int init_cpu_server_client_mgr();

void free_cpu_server_client_mgr();

int add_new_client(int pid, int xp_fd);

cricket_client* get_client(int xp_fd);

cricket_client* get_client_by_pid(int pid);

int remove_client(int xp_fd);

int remove_client_by_pid(int pid);

#endif // __FLYT_CPU_SERVER_CLIENT_MGR_H