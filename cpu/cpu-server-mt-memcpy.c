#include <stdint.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "cpu-server-mt-memcpy.h"
#include "log.h"
#include "resource-mg.h"
#include "cpu-server-client-mgr.h"

struct copy_thread_args {
    mt_memcpy_server_t *server;
    int socket;
    uint32_t thread_id;
    pthread_t thread;
};

static void* mt_memcpy_copy_thread(void* targs)
{
    struct copy_thread_args* args = (struct copy_thread_args*)targs;

    cricket_client *client = get_client(args->server->client_fd);

    if (client == NULL) {
        LOGE(LOG_ERROR, "oob: failed to get client.");
        return (void*)1;
    }

    int iret;
    size_t ret = 1;
    if (args->server->dir == MT_MEMCPY_HTOD) {
        if ((iret = oob_receive_s(args->socket, &args->thread_id, 
                                  sizeof(uint32_t))) != sizeof(uint32_t)) {
            LOGE(LOG_ERROR, "oob: failed to receive memory: received %d bytes: %s", iret, strerror(errno));
            return (void*)ret;
        }
    }
    size_t mem_per_thread = (args->server->mem_size / (size_t)args->server->thread_num);
    size_t mem_offset = (size_t)args->thread_id * mem_per_thread;
    size_t mem_this_thread = mem_per_thread;
    if (args->thread_id == args->server->thread_num - 1) {
        mem_this_thread = args->server->mem_size - mem_offset;
    }
    if (args->server->dir == MT_MEMCPY_HTOD) {
        if ((iret = oob_receive_s(args->socket, args->server->mem_ptr+mem_offset, 
                                  mem_this_thread)) != mem_this_thread) {
            LOGE(LOG_ERROR, "oob: failed to receive memory: received %d bytes: %s", iret, strerror(errno));
            return (void*)ret;
        }

        void *dev_ptr = resource_map_get_addr(client->gpu_mem, args->server->dev_ptr);
        if (dev_ptr == NULL) {
            LOGE(LOG_ERROR, "oob: failed to get device pointer.");
            return (void*)ret;
        }


        cudaError_t res = cudaMemcpy(
          dev_ptr + mem_offset,
          args->server->mem_ptr+mem_offset,
          mem_this_thread,
          cudaMemcpyHostToDevice);
        if (res != cudaSuccess) {
            LOGE(LOG_ERROR, "oob: failed to copy memory: %s", cudaGetErrorString(res));
        }
    } else if (args->server->dir == MT_MEMCPY_DTOH) {

        void *dev_ptr = resource_map_get_addr(client->gpu_mem, args->server->dev_ptr);
        if (dev_ptr == NULL) {
            LOGE(LOG_ERROR, "oob: failed to get device pointer.");
            return (void*)ret;
        }

        cudaError_t res = cudaMemcpy(
            args->server->mem_ptr+mem_offset,
            dev_ptr+mem_offset,
            mem_this_thread,
            cudaMemcpyDeviceToHost);
        if (res != cudaSuccess) {
            LOGE(LOG_ERROR, "oob: failed to copy memory: %s", cudaGetErrorString(res));
            return (void*)ret;
        }
    
        if ((iret = oob_send_s(args->socket,
                               &args->thread_id,
                               sizeof(uint32_t))) != sizeof(uint32_t)) {
            LOGE(LOG_ERROR, "oob: failed to send memory: sent %d bytes: %s", iret, strerror(errno));
            return (void*)ret;
        }
        if ((iret = oob_send_s(args->socket,
                               args->server->mem_ptr+mem_offset,
                               mem_this_thread)) != mem_this_thread) {
            LOGE(LOG_ERROR, "oob: failed to send memory: sent %d bytes: %s", iret, strerror(errno));
            return (void*)ret;
        }
    }
    close(args->socket);
    ret = 0;
    return (void*)ret;
}

static void* mt_memcpy_listener_thread(void* targs)
{
    mt_memcpy_server_t *args = (mt_memcpy_server_t*)targs;
    size_t ret = 1;
    if (oob_init_listener_socket(&args->oob, args->port) != 0) {
        LOGE(LOG_ERROR, "oob: failed to initialize listener socket.");
        return (void*)ret;
    }
    args->port = args->oob.port;
    //LOGE(LOG_DBG(1), "oob: listening on port %d.", args->port);
    int iret = pthread_barrier_wait(&args->barrier);
    if (iret != 0 && iret != PTHREAD_BARRIER_SERIAL_THREAD) {
        LOGE(LOG_ERROR, "oob: failed to wait for barrier.");
        return (void*)ret;
    }
    struct copy_thread_args *copy_args;
    if ((copy_args = malloc(sizeof(struct copy_thread_args) * args->thread_num)) == NULL) {
        return (void*)ret;
    }
    for (int i=0; i < args->thread_num; i++) {
        copy_args[i].server = args;
        copy_args[i].thread_id = i;
        if (oob_init_listener_accept(&args->oob, &copy_args[i].socket) != 0) {
            LOGE(LOG_ERROR, "mt_memcpy: failed to accept connection.");
            return (void*)ret;
        }

        if ((iret = pthread_create(&copy_args[i].thread,
                                   NULL,
                                   mt_memcpy_copy_thread,
                                   &copy_args[i])) != 0) {
            LOGE(LOG_ERROR, "mt_memcpy: failed to create copy thread: %s", strerror(errno));
            return (void*)ret;
        }
    }
    

    for (int i=0; i < args->thread_num; i++) {
        pthread_join(copy_args[i].thread, (void**)&ret);
        if (ret != 0) {
            LOGE(LOG_ERROR, "mt_memcpy: failed to copy memory.");
        }
    }
    //cudaFreeHost(args->mem_ptr);
    free(args->mem_ptr);
    if (oob_close(&args->oob) != 0) {
        LOGE(LOG_ERROR, "oob: failed to close connection.");
        return (void*)ret;
    }
    ret = 0;
    free (copy_args);
    return (void*)ret;
}

int mt_memcpy_init_server(mt_memcpy_server_t *server, void* dev_ptr, size_t size, enum mt_memcpy_direction dir, int thread_num, int client_fd)
{
    cudaError_t cudaRes;
    int ret = 1;
    if (server == NULL) {
        return ret;
    }
    server->dev_ptr = dev_ptr;
    server->port = 0;
    server->mem_size = size;
    server->dir = dir;
    server->thread_num = thread_num;
    server->client_fd = client_fd;
    if (pthread_barrier_init(&server->barrier, NULL, 2) != 0) {
        LOGE(LOG_ERROR, "oob: failed to initialize barrier.");
        return ret;
    }

    if (pthread_create(&server->server_thread, NULL, mt_memcpy_listener_thread, server) != 0) {
        LOGE(LOG_ERROR, "oob: failed to create listener thread.");
        goto cleanup;
    }

    if ((server->mem_ptr = malloc(server->mem_size)) == NULL) {
        LOGE(LOG_ERROR, "oob: failed to allocate memory.");
        pthread_cancel(server->server_thread);
        goto cleanup;
    }
    /*if ((cudaRes = cudaHostAlloc(&server->mem_ptr, server->mem_size, 0)) != cudaSuccess) {
        LOGE(LOG_ERROR, "cudaHostAlloc failed: %d.", cudaRes);
        pthread_cancel(server->server_thread);
        goto cleanup;
    }*/

    int bret = pthread_barrier_wait(&server->barrier);
    if (bret != 0 && bret != PTHREAD_BARRIER_SERIAL_THREAD) {
        LOGE(LOG_ERROR, "oob: failed to wait for barrier.");
        pthread_cancel(server->server_thread);
        goto cleanup;
    }
    ret = 0;
 cleanup:
    pthread_barrier_destroy(&server->barrier);
    return ret;
}

int mt_memcpy_sync_server(mt_memcpy_server_t *server)
{
    size_t ret;
    if (pthread_join(server->server_thread, (void**)&ret) != 0) {
        LOGE(LOG_ERROR, "oob: failed to join listener thread.");
        return 1;
    }
    return ret;
}