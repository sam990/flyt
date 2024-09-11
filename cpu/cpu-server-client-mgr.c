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
#include "cpu-server-ivshmem.h"

#define STR_DUMP_SIZE 128

static pthread_mutex_t client_mgr_mutex = PTHREAD_MUTEX_INITIALIZER;


//static resource_mg pid_to_xp_fd;
static resource_mg xp_fd_to_client;
static resource_mg restored_clients;


static int freeResources(cricket_client* client);

int init_cpu_server_client_mgr() {
    int ret = 0;
    /*
    ret = resource_mg_init(&pid_to_xp_fd, 0);
    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to initialize pid_to_xp_fd resource manager");
        return ret;
    }
    */
    ret = resource_mg_init(&xp_fd_to_client, 0); // list of fds tcreated on server for each client process
    if (ret != 0) {
        //resource_mg_free(&pid_to_xp_fd);
        LOGE(LOG_ERROR, "Failed to initialize xp_fd_to_client resource manager");
        return ret;
    }
    ret = resource_mg_init(&restored_clients, 0);
    if (ret != 0) {
        //resource_mg_free(&pid_to_xp_fd);
        resource_mg_free(&xp_fd_to_client);
        LOGE(LOG_ERROR, "Failed to initialize restored_clients resource manager");
        return ret;
    }

    return ret;
}

void free_cpu_server_client_mgr() {
    //resource_mg_free(&pid_to_xp_fd);
    resource_mg_free(&xp_fd_to_client);
    resource_mg_free(&restored_clients);
}

cricket_client* create_client(int pid, ivshmem_svc_ctx *_ctx) {

    cricket_client* client = (cricket_client*)malloc(sizeof(cricket_client));
    if (client == NULL) {
        LOGE(LOG_ERROR, "Failed to allocate memory for new client");
        return NULL;
    }
    client->pid = pid;
    client->gpu_mem = init_resource_map(INIT_MEM_SLOTS); // 4096 allocs per client, but extends dynamically
    if (client->gpu_mem == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize gpu_mem resource map for new client");
        free(client);
        return NULL;
    }
    client->default_stream = NULL; // create a default stream

    cudaError_t err = cudaStreamCreateWithFlags((cudaStream_t *)&client->default_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        LOGE(LOG_ERROR, "Failed to create default stream for new client: %s", cudaGetErrorString(err));
        free_resource_map(client->gpu_mem);
        free(client);
        return NULL;
    }

    client->custom_streams = init_resource_map(INIT_STREAM_SLOTS);
    if (client->custom_streams == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map for new client");
        free_resource_map(client->gpu_mem);
        free(client);
        return NULL;
    }

    // ivshmem added.
    client->ivshmem_ctx = _ctx;

    resource_mg_init(&client->modules, 0);
    resource_mg_init(&client->functions, 0);
    resource_mg_init(&client->vars, 0);
    LOGE(LOG_INFO, "added client for pid %d\n", pid);

    return client;

}

// create a new client
// add <&fd, &client> to xp_fd_to_client map.
int add_new_client(int pid, int xp_fd, ivshmem_svc_ctx *_ctx) {

    cricket_client* client = create_client(pid, _ctx);
    if (client == NULL) {
        LOGE(LOG_ERROR, "Failed to create new client for pid %d", pid);
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);

    //int ret = resource_mg_add_sorted(&pid_to_xp_fd, (void *)(long)pid, (void *)(long)xp_fd);
    //ret |= resource_mg_add_sorted(&xp_fd_to_client, (void *)(long)xp_fd, client);
    int ret = resource_mg_add_sorted(&xp_fd_to_client, (void *)(long)xp_fd, client);


    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to add new client to resource managers");
        free_resource_map(client->gpu_mem);
        free_resource_map(client->custom_streams);
        resource_mg_free(&client->modules);
        resource_mg_free(&client->functions);
        resource_mg_free(&client->vars);
        free(client);
        return -1;
    }
    pthread_mutex_unlock(&client_mgr_mutex);

    return (ret != 0)? -1:0;
}

int add_restored_client(cricket_client *client) {
    pthread_mutex_lock(&client_mgr_mutex);
    int ret = resource_mg_add_sorted(&restored_clients, (void *)(long)client->pid, (void *)client);
    pthread_mutex_unlock(&client_mgr_mutex);
    return ret;
}

int move_restored_client(int pid, int xp_fd) {
    cricket_client* client = (cricket_client*)resource_mg_get_default(&restored_clients, (void *)(long)pid, NULL);
    
    if (client == NULL) {
        LOGE(LOG_ERROR, "Client with pid %d not found in restored clients", pid);
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);
    //int ret = resource_mg_add_sorted(&pid_to_xp_fd, (void *)(long)client->pid, (void *)(long)xp_fd);
    //ret &= resource_mg_add_sorted(&xp_fd_to_client, (void *)(long)xp_fd, client);
    int ret = resource_mg_add_sorted(&xp_fd_to_client, (void *)(long)xp_fd, client);
    pthread_mutex_unlock(&client_mgr_mutex);

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to add restored client to resource managers");
        return -1;
    }

    return resource_mg_remove(&restored_clients, (void *)(long)pid);
}


inline cricket_client* get_client(int xp_fd) {
    // pass map-to-be-looked-up, address of client fd.
	cricket_client *ret = (cricket_client*)resource_mg_get_default(&xp_fd_to_client, (void *)(long)xp_fd, NULL);
	if(ret == NULL) {
	    LOGE(LOG_ERROR, "get client for %d\n", xp_fd);
	    resource_mg_print(&xp_fd_to_client);
	}
	return ret;
}

/*
inline cricket_client* get_client_by_pid(int pid) {
    int xp_fd = (int)(long)resource_mg_get_default(&pid_to_xp_fd, (void *)(long)pid, (void*)-1ll);
    if (xp_fd == -1) {
        return NULL;
    }
    return get_client(xp_fd);
}
*/

int remove_client_ptr(cricket_client* client) {
    if (client == NULL) {
        LOGE(LOG_ERROR, "Client is null");
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);
    LOGE(LOG_INFO, "removing client ptr from %d \n", client->pid);
    //resource_mg_remove(&pid_to_xp_fd, (void *)(long)client->pid);
    
    // need to free gpu resources and custom streams
    freeResources(client);
    

    // free client
    free_resource_map(client->gpu_mem);
    free_resource_map(client->custom_streams);

    pthread_mutex_lock(&client->modules.mutex);
    pthread_mutex_lock(&client->modules.map_res.mutex);
    for (size_t i = 0; i < client->modules.map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&client->modules.map_res, i);
	if(elem != NULL) {

            addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

            free_module_data(pair);
	}
    }
    pthread_mutex_unlock(&client->modules.map_res.mutex);
    pthread_mutex_unlock(&client->modules.mutex);

    pthread_mutex_lock(&client->vars.mutex);
    pthread_mutex_lock(&client->vars.map_res.mutex);
    for (size_t i = 0; i < client->vars.map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&client->vars.map_res, i);
	if(elem != NULL) {

            addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
        
            free_variable_data(pair);
	}
    }
    pthread_mutex_unlock(&client->vars.map_res.mutex);
    pthread_mutex_unlock(&client->vars.mutex);

    pthread_mutex_lock(&client->functions.mutex);
    pthread_mutex_lock(&client->functions.map_res.mutex);
    for (size_t i = 0; i < client->functions.map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&client->functions.map_res, i);
	if(elem != NULL) {

            addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
        
            free_function_data(pair);
	}
    }
    pthread_mutex_unlock(&client->functions.map_res.mutex);
    pthread_mutex_unlock(&client->functions.mutex);

    resource_mg_free(&client->modules);
    resource_mg_free(&client->vars);
    resource_mg_free(&client->functions);

    pthread_mutex_unlock(&client_mgr_mutex);
    free(client);
    return 0;
}

int remove_client(int xp_fd) {
    cricket_client* client = get_client(xp_fd);
    if (client == NULL) {
        LOGE(LOG_ERROR, "Client with xp_fd %d not found", xp_fd);
        return -1;
    }

    LOGE(LOG_INFO, "Client with xp_fd %d being removed", xp_fd);
    int ret = remove_client_ptr(client);

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to remove client with xp_fd %d", xp_fd);
        return -1;
    }

    return resource_mg_remove(&xp_fd_to_client, (void *)(long)xp_fd);
}

static int freeResources(cricket_client* client) {
    PRIMARY_CTX_RETAIN;
    
    resource_map_iter *mem_iter = resource_map_init_iter(client->gpu_mem);

    if (mem_iter == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize gpu_mem resource map iterator");
        return -1;
    }

    uint64_t mem_idx;
    while ((mem_idx = resource_map_iter_next(mem_iter)) != 0) {
        cudaFree(resource_map_get_addr(client->gpu_mem, (void *)mem_idx));
    }

    resource_map_free_iter(mem_iter);

    resource_map_iter *stream_iter = resource_map_init_iter(client->custom_streams);
    
    if (stream_iter == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map iterator");
        return -1;
    }

    uint64_t stream_idx;
    while ((stream_idx = resource_map_iter_next(stream_iter)) != 0) {
        cudaStreamDestroy((cudaStream_t)resource_map_get_addr(client->custom_streams, (void *)stream_idx));
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

    resource_mg_map_elem *elem;
    if(resource_mg_get_element_at(&xp_fd_to_client, TRUE, *iter, (void **)&elem) != 0) {
	    return NULL;
    }
    *iter += 1;
    return (cricket_client *)elem->cuda_address;
}


cricket_client* get_next_restored_client(cricket_client_iter* iter) {
    
    if (iter == NULL) {
        return NULL;
    }

    resource_mg_map_elem *elem;
    if(resource_mg_get_element_at(&restored_clients, TRUE, *iter, (void **)&elem) != 0) {
	    return NULL;
    }
    *iter += 1;
    return (cricket_client *)elem->cuda_address;
}

void free_variable_data(addr_data_pair_t *pair) {
    switch (pair->reg_data.type) {
        case MODULE_GET_GLOBAL:
            var_register_args_t *data = (var_register_args_t *)pair->reg_data.data;
            free(data->deviceName); // it is an string arg
            if (data->data) {
                free(data->data);
            }
            free(pair->reg_data.data);
            break;
        default:
            LOGE(LOG_ERROR, "Invalid variable data type: %d", pair->reg_data.type);
            break;
    }
    free(pair);
}

void free_function_data(addr_data_pair_t *pair) {

    switch (pair->reg_data.type) {
        case MODULE_GET_FUNCTION:
            rpc_cumodulegetfunction_1_argument *data = (rpc_cumodulegetfunction_1_argument *)pair->reg_data.data;
            free(data->arg2); // it is an string arg
            free(pair->reg_data.data);
            break;
        default:
            LOGE(LOG_ERROR, "Invalid function data type: %d", pair->reg_data.type);
            break;
    }
    free(pair);
}


void free_module_data(addr_data_pair_t *pair) {
    
    switch (pair->reg_data.type) {
        case MODULE_LOAD:
            free(pair->reg_data.data);
            break;
        case MODULE_LOAD_DATA:
            mem_data* data = (mem_data *)pair->reg_data.data;
            free(data->mem_data_val);
            free(pair->reg_data.data);
            break;
        default:
            LOGE(LOG_ERROR, "Invalid module data type: %d", pair->reg_data.type);
            break;
    }
    free(pair);
}

int fetch_variable_data_to_host(void) {

    cricket_client_iter iter = get_client_iter();

    cricket_client* client;

    while ((client = get_next_client(&iter)) != NULL) {

        pthread_mutex_lock(&client->vars.mutex);
        pthread_mutex_lock(&client->vars.map_res.mutex);
        for (size_t i = 0; i < client->vars.map_res.length; i++) {
            resource_mg_map_elem *elem = list_get(&client->vars.map_res, i);
	    if(elem != NULL) {

                addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
            
                var_register_args_t *data = (var_register_args_t *)pair->reg_data.data;

                if (data->data == NULL) {
                    data->data = malloc(data->size);
                }

                if (cudaMemcpyFromSymbol(data->data, pair->addr, data->size, 0, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    LOGE(LOG_ERROR, "Failed to copy variable data to host");
                    pthread_mutex_unlock(&client->vars.map_res.mutex);
                    pthread_mutex_unlock(&client->vars.mutex);
                    return -1;
                }
	    }
        }
        pthread_mutex_unlock(&client->vars.map_res.mutex);
        pthread_mutex_unlock(&client->vars.mutex);
    }
    return 0;
}

int dump_module_data(resource_mg_map_elem *elem, FILE* fp) {

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    fwrite(elem, sizeof(resource_mg_map_elem), 1, fp);
    fwrite(pair, sizeof(addr_data_pair_t), 1, fp);

    switch (pair->reg_data.type) {
        case MODULE_LOAD:
            {
                char* data = (char *)pair->reg_data.data;
                fwrite(data, sizeof(char), pair->reg_data.size, fp);
                break;
            }
        case MODULE_LOAD_DATA:
            {
                mem_data* data = (mem_data *)pair->reg_data.data;
                fwrite(data, sizeof(mem_data), 1, fp);
                fwrite(data->mem_data_val, data->mem_data_len, 1, fp);
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid module data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int load_module_data(resource_mg_map_elem *elem, FILE* fp) {
    size_t readsz;
    readsz = fread(elem, sizeof(resource_mg_map_elem), 1, fp);

    if (readsz != 1) {
        return -1;
    }

    elem->cuda_address = malloc(sizeof(addr_data_pair_t));
    readsz = fread(elem->cuda_address, sizeof(addr_data_pair_t), 1, fp);

    if (readsz != 1) {
        free(elem->cuda_address);
        return -1;
    }

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    switch (pair->reg_data.type) {
        case MODULE_LOAD:
            {
                char* data = (char *)malloc(pair->reg_data.size);
                readsz = fread(data, sizeof(char), pair->reg_data.size, fp);
                if (readsz != pair->reg_data.size) {
                    free(data);
                    free(pair);
                    return -1;
                }
                pair->reg_data.data = data;
                break;
            }
        case MODULE_LOAD_DATA:
            {
                mem_data* data = (mem_data *)malloc(sizeof(mem_data));
                readsz = fread(data, sizeof(mem_data), 1, fp);

                if (readsz != 1) {
                    free(data);
                    free(pair);
                    return -1;
                }

                data->mem_data_val = (char *)malloc(data->mem_data_len);
                readsz = fread(data->mem_data_val, data->mem_data_len, 1, fp);
                
                if (readsz != 1) {
                    free(data->mem_data_val);
                    free(data);
                    free(pair);
                    return -1;
                }

                pair->reg_data.data = data;
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid module data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int dump_variable_data(resource_mg_map_elem *elem, FILE* fp) {

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    fwrite(elem, sizeof(resource_mg_map_elem), 1, fp);
    fwrite(pair, sizeof(addr_data_pair_t), 1, fp);

    char str_dump[STR_DUMP_SIZE];

    switch (pair->reg_data.type) {
        case MODULE_GET_GLOBAL:
            {
                var_register_args_t *data = (var_register_args_t *)pair->reg_data.data;
                fwrite(data, sizeof(var_register_args_t), 1, fp);
                strncpy(str_dump, data->deviceName, STR_DUMP_SIZE);
                fwrite(str_dump, sizeof(char), STR_DUMP_SIZE, fp);

                size_t var_size = data->size;
                char *var_data = (char *)malloc(var_size);

                cudaMemcpyFromSymbol(var_data, pair->addr, var_size, 0, cudaMemcpyDeviceToHost);

                fwrite(var_data, var_size, 1, fp);

                free(var_data);

                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid variable data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int load_variable_data(resource_mg_map_elem *elem, FILE *fp) {
    size_t readsz;
    readsz = fread(elem, sizeof(resource_mg_map_elem), 1, fp);

    if (readsz != 1) {
        return -1;
    }

    elem->cuda_address = malloc(sizeof(addr_data_pair_t));
    readsz = fread(elem->cuda_address, sizeof(addr_data_pair_t), 1, fp);

    if (readsz != 1) {
        free(elem->cuda_address);
        return -1;
    }

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    switch (pair->reg_data.type) {
        case MODULE_GET_GLOBAL:
            {
                var_register_args_t *data = (var_register_args_t *)malloc(sizeof(var_register_args_t));
                readsz = fread(data, sizeof(var_register_args_t), 1, fp);

                if (readsz != 1) {
                    free(data);
                    free(pair);
                    return -1;
                }

                data->deviceName = (char *)malloc(STR_DUMP_SIZE);
                readsz = fread(data->deviceName, sizeof(char), STR_DUMP_SIZE, fp);
                
                if (readsz != STR_DUMP_SIZE) {
                    free(data->deviceName);
                    free(data);
                    free(pair);
                    return -1;
                }

                data->data = malloc(data->size);
                readsz = fread(data->data, data->size, 1, fp);

                if (readsz != 1) {
                    free(data->data);
                    free(data->deviceName);
                    free(data);
                    free(pair);
                    return -1;
                }

                

                pair->reg_data.data = data;
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid variable data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int dump_function_data(resource_mg_map_elem *elem, FILE* fp) {

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    fwrite(elem, sizeof(resource_mg_map_elem), 1, fp);
    fwrite(pair, sizeof(addr_data_pair_t), 1, fp);

    char str_dump[STR_DUMP_SIZE];

    switch (pair->reg_data.type) {
        case MODULE_GET_FUNCTION:
            {
                rpc_cumodulegetfunction_1_argument *data = (rpc_cumodulegetfunction_1_argument *)pair->reg_data.data;
                fwrite(data, sizeof(rpc_cumodulegetfunction_1_argument), 1, fp);
                strncpy(str_dump, data->arg2, STR_DUMP_SIZE);
                fwrite(str_dump, sizeof(char), STR_DUMP_SIZE, fp);
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid function data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int load_function_data(resource_mg_map_elem *elem, FILE *fp) {
    size_t readsz;
    readsz = fread(elem, sizeof(resource_mg_map_elem), 1, fp);

    if (readsz != 1) {
        return -1;
    }

    elem->cuda_address = malloc(sizeof(addr_data_pair_t));
    readsz = fread(elem->cuda_address, sizeof(addr_data_pair_t), 1, fp);

    if (readsz != 1) {
        free(elem->cuda_address);
        return -1;
    }

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    switch (pair->reg_data.type) {
        case MODULE_GET_FUNCTION:
            {
                rpc_cumodulegetfunction_1_argument *data = (rpc_cumodulegetfunction_1_argument *)malloc(sizeof(rpc_cumodulegetfunction_1_argument));
                readsz = fread(data, sizeof(rpc_cumodulegetfunction_1_argument), 1, fp);

                if (readsz != 1) {
                    free(data);
                    free(pair);
                    return -1;
                }

                data->arg2 = (char *)malloc(STR_DUMP_SIZE);
                readsz = fread(data->arg2, sizeof(char), STR_DUMP_SIZE, fp);
                
                if (readsz != STR_DUMP_SIZE) {
                    free(data->arg2);
                    free(data);
                    free(pair);
                    return -1;
                }

                pair->reg_data.data = data;
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid function data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}
