#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>

#include "log.h"
#include "resource-map.h"
#include "resource-mg.h"
#include "cpu-server-client-mgr.h"
#include "cpu-server-resource-controller.h"
#include "cpu_rpc_prot.h"

static pthread_mutex_t client_mgr_mutex = PTHREAD_MUTEX_INITIALIZER;


static resource_mg pid_to_xp_fd;
static resource_mg xp_fd_to_client;

int init_cpu_server_client_mgr() {
    int ret = 0;
    ret = resource_mg_init(&pid_to_xp_fd, 0);
    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to initialize pid_to_xp_fd resource manager");
        return ret;
    }
    ret = resource_mg_init(&xp_fd_to_client, 0);
    if (ret != 0) {
        resource_mg_free(&pid_to_xp_fd);
        LOGE(LOG_ERROR, "Failed to initialize xp_fd_to_client resource manager");
        return ret;
    }
    return ret;
}

void free_cpu_server_client_mgr() {
    resource_mg_free(&pid_to_xp_fd);
    resource_mg_free(&xp_fd_to_client);
}

int add_new_client(int pid, int xp_fd) {

    cricket_client* client = (cricket_client*)malloc(sizeof(cricket_client));
    if (client == NULL) {
        LOGE(LOG_ERROR, "Failed to allocate memory for new client");
        return -1;
    }
    client->pid = pid;
    client->gpu_mem = init_resource_map(INIT_MEM_SLOTS);
    if (client->gpu_mem == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize gpu_mem resource map for new client");
        free(client);
        return -1;
    }
    client->default_stream = NULL; // create a default stream

    cudaError_t err = cudaStreamCreate(&client->default_stream);
    if (err != cudaSuccess) {
        LOGE(LOG_ERROR, "Failed to create default stream for new client: %s", cudaGetErrorString(err));
        free_resource_map(client->gpu_mem);
        free(client);
        return -1;
    }

    client->custom_streams = init_resource_map(INIT_STREAM_SLOTS);
    if (client->custom_streams == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map for new client");
        free_resource_map(client->gpu_mem);
        free(client);
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);

    int ret = resource_mg_add_sorted(&pid_to_xp_fd, (void *)pid, (void *)xp_fd);
    ret &= resource_mg_add_sorted(&xp_fd_to_client, (void *)xp_fd, client);

    pthread_mutex_unlock(&client_mgr_mutex);

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to add new client to resource managers");
        free_resource_map(client->gpu_mem);
        free_resource_map(client->custom_streams);
        free(client);
        return -1;
    }

    resource_mg_init(&client->modules, 0);
    client->functions = init_resource_map(INIT_FUNCTION_SLOTS);
    resource_mg_init(&client->vars, 0);

    return 0;
}

inline cricket_client* get_client(int xp_fd) {
    return (cricket_client*)resource_mg_get_default(&xp_fd_to_client, (void *)xp_fd, NULL);
}

inline cricket_client* get_client_by_pid(int pid) {
    int xp_fd = resource_mg_get_default(&pid_to_xp_fd, (void *)pid, -1);
    if (xp_fd == -1) {
        return NULL;
    }
    return get_client(xp_fd);
}

int remove_client(int xp_fd) {
    cricket_client* client = get_client(xp_fd);
    if (client == NULL) {
        LOGE(LOG_ERROR, "Client with xp_fd %d not found", xp_fd);
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);
    resource_mg_remove(&pid_to_xp_fd, (void *)client->pid);
    pthread_mutex_unlock(&client_mgr_mutex);
    
    // need to free gpu resources and custom streams
    freeResources(client);
    

    // free client
    free_resource_map(client->gpu_mem);
    free_resource_map(client->custom_streams);
    free_resource_map(client->functions);

    for (size_t i = 0; i < client->modules.map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&client->modules.map_res, i);

        addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

        mem_data* data = (mem_data *)pair->reg_data;
        free(data->mem_data_val);
        free(pair->reg_data);
        free(pair);
    }

    for (size_t i = 0; i < client->vars.map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&client->vars.map_res, i);

        addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
        rpc_register_var_1_argument* data = (rpc_register_var_1_argument *)pair->reg_data;
        
        free(data->arg4); // it is an string arg
        free(pair->reg_data);
        free(pair);
    }

    resource_mg_free(&client->modules);
    resource_mg_free(&client->vars);

    free(client);
    return resource_mg_remove(&xp_fd_to_client, (void *)xp_fd);
}



int freeResources(cricket_client* client) {
    PRIMARY_CTX_RETAIN;
    
    resource_map_iter *mem_iter = resource_map_init_iter(client->gpu_mem);

    if (mem_iter == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize gpu_mem resource map iterator");
        return -1;
    }

    uint64_t mem_idx;
    while ((mem_idx = resource_map_iter_next(mem_iter)) != 0) {
        cudaFree(resource_map_get(client->gpu_mem, (void *)mem_idx));
    }

    resource_map_free_iter(mem_iter);

    resource_map_iter *stream_iter = resource_map_init_iter(client->custom_streams);
    
    if (stream_iter == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map iterator");
        return -1;
    }

    uint64_t stream_idx;
    while ((stream_idx = resource_map_iter_next(stream_iter)) != 0) {
        cudaStreamDestroy(resource_map_get(client->custom_streams, (void *)stream_idx));
    }

    resource_map_free_iter(stream_iter);

    cudaStreamDestroy(client->default_stream);
    

    PRIMARY_CTX_RELEASE;
}


cricket_client_iter get_client_iter() {
    return 0;
}

cricket_client* get_next_client(cricket_client_iter* iter) {
    
    if (iter == NULL) {
        return NULL;
    }

    if (*iter >= xp_fd_to_client.map_res.length) {
        return NULL;
    }

    resource_mg_map_elem *elem = list_get(&xp_fd_to_client.map_res, *iter);
    *iter += 1;
    return (cricket_client *)elem->cuda_address;
}