#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "log.h"
#include "resource-map.h"
#include "resource-mg.h"
#include "cpu-server-client-mgr.h"



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

    client->custom_streams = init_resource_map(INIT_STREAM_SLOTS);
    if (client->custom_streams == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map for new client");
        free_resource_map(client->gpu_mem);
        free(client);
        return -1;
    }

    int ret = resource_mg_add_sorted(&pid_to_xp_fd, (void *)pid, (void *)xp_fd);
    ret &= resource_mg_add_sorted(&xp_fd_to_client, (void *)xp_fd, client);

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to add new client to resource managers");
        free_resource_map(client->gpu_mem);
        free_resource_map(client->custom_streams);
        free(client);
        return -1;
    }

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

    resource_mg_remove(&pid_to_xp_fd, (void *)client->pid);

    // need to free gpu resources and custom streams


    // free client
    free_resource_map(client->gpu_mem);
    free_resource_map(client->custom_streams);
    free(client);
    return resource_mg_remove(&xp_fd_to_client, (void *)xp_fd);
}

