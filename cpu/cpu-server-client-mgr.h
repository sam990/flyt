# ifndef __FLYT_CPU_SERVER_CLIENT_MGR_H
# define __FLYT_CPU_SERVER_CLIENT_MGR_H

#include <stdio.h>
#include "resource-mg.h"
#include "resource-map.h"

#define INIT_MEM_SLOTS 4096
#define INIT_STREAM_SLOTS 16
#define INIT_MODULE_SLOTS 8
#define INIT_FUNCTION_SLOTS 128
#define INIT_VAR_SLOTS 128





enum MODULE_LOAD_TYPE {
    MODULE_LOAD_DATA = 0,
    MODULE_LOAD = 1,
};

enum MODULE_GET_FUNCTION_TYPE {
    MODULE_GET_FUNCTION = 0
};

enum MODULE_GET_GLOBAL_TYPE {
    MODULE_GET_GLOBAL = 0
};

enum MEM_ALLOC_TYPE {
    MEM_ALLOC_TYPE_DEFAULT = 0,
    MEM_ALLOC_TYPE_3D = 1,
    MEM_ALLOC_TYPE_PITCH = 2,
};

typedef struct __mem_alloc_args {
    enum MEM_ALLOC_TYPE type;
    long long size;
    size_t depth;
    size_t height;
    size_t width;
    size_t pitch;
    long long arg6; 
} mem_alloc_args_t;

typedef struct __var_register_args {
    void* module;
    char* deviceName;
    size_t size;
    void* data;
} var_register_args_t;


enum STREAM_CREATE_TYPE {
    STREAM_CREATE_TYPE_DEFAULT = 0,
    STREAM_CREATE_TYPE_FLAGS = 1,
    STREAM_CREATE_TYPE_PRIORITY = 2,
};

typedef struct __stream_create_args {
    enum STREAM_CREATE_TYPE type;
    int flags;
    int priority;
} stream_create_args_t;



typedef struct __addr_data_pair {
    void* addr;
    struct {
        int type;
        size_t size;
        void* data;
    } reg_data;
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

int remove_client_ptr(cricket_client *client);

int remove_client(int xp_fd);

cricket_client_iter get_client_iter();

cricket_client *get_next_client(cricket_client_iter *iter);

cricket_client *get_next_restored_client(cricket_client_iter *iter);

void free_variable_data(addr_data_pair_t *pair);

void free_function_data(addr_data_pair_t *pair);

void free_module_data(addr_data_pair_t *pair);

int fetch_variable_data_to_host(void);

int dump_module_data(resource_mg_map_elem *elem, FILE *fp);

int load_module_data(resource_mg_map_elem *elem, FILE *fp);

int dump_variable_data(resource_mg_map_elem *elem, FILE *fp);

int load_variable_data(resource_mg_map_elem *elem, FILE *fp);

int dump_function_data(resource_mg_map_elem *elem, FILE *fp);

int load_function_data(resource_mg_map_elem *elem, FILE *fp);

int remove_client_by_pid(int pid);



#define GET_CLIENT(result) cricket_client *client = get_client(rqstp->rq_xprt->xp_fd); \
    if (client == NULL) { \
        LOGE(LOG_ERROR, "error getting client"); \
        result= cudaErrorInvalidValue; \
        GSCHED_RELEASE; \
        return 1; \
    }


#define GET_STREAM(stream_ptr, stream, result) \
    if (stream == 0) { \
        stream_ptr = client->default_stream; \
    } else if (!resource_map_contains(client->custom_streams, (void*)stream)) { \
        LOGE(LOG_ERROR, "stream not found in custom streams"); \
        result = cudaErrorInvalidValue; \
        GSCHED_RELEASE; \
        return 1; \
    } else { \
        stream_ptr = resource_map_get_addr(client->custom_streams, (void*)stream); \
    }

#define GET_MEMORY(mem_ptr, client_addr, result) \
    if (!resource_map_contains(client->gpu_mem, (void*)client_addr)) { \
        LOGE(LOG_ERROR, "memory not found in gpu_mem"); \
        result = cudaErrorInvalidValue; \
        GSCHED_RELEASE; \
        return 1; \
    } \
    mem_ptr = resource_map_get_addr(client->gpu_mem, (void*)client_addr);

#define GET_VARIABLE(var_ptr, client_addr, result) \
    if ((var_ptr = resource_mg_get_or_null(&client->vars, (void *)client_addr)) == NULL) { \
        LOGE(LOG_ERROR, "variable not found in gpu_vars"); \
        result = cudaErrorInvalidValue; \
        GSCHED_RELEASE; \
        return 1; \
    }


#endif // __FLYT_CPU_SERVER_CLIENT_MGR_H