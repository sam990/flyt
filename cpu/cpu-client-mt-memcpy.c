#include <stdint.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "cpu-client-mt-memcpy.h"
#include "log.h"
#include "resource-mg.h"


struct client_args {
    struct addrinfo *addr;
    void* host_ptr;
    size_t size;
    enum mt_memcpy_direction dir;
    int thread_num;
};

struct client_thread_args {
    struct client_args* client;
    uint32_t thread_id;
    pthread_t thread;
};

static void* mt_memcpy_client_thread(void* arg)
{
    size_t ret = 1;
    int sock;
    struct client_thread_args *args = (struct client_thread_args*)arg;
    if (oob_init_sender_s(&sock, args->client->addr) != 0) {
        LOGE(LOG_ERROR, "oob_init_sender failed");
        goto cleanup;
    }
    if (args->client->dir == MT_MEMCPY_DTOH) {
        if (oob_receive_s(sock, &args->thread_id, sizeof(uint32_t)) != sizeof(uint32_t)) {
            LOGE(LOG_ERROR, "oob_send failed");
            goto cleanup;
        }
    }

    size_t mem_per_thread = (args->client->size / (size_t)args->client->thread_num);
    size_t mem_offset = (size_t)args->thread_id * mem_per_thread;
    size_t mem_this_thread = mem_per_thread;
    if (args->thread_id == args->client->thread_num - 1) {
        mem_this_thread = args->client->size - mem_offset;
    }
    if (args->client->dir == MT_MEMCPY_HTOD) {
        if (oob_send_s(sock, &args->thread_id, sizeof(uint32_t)) != sizeof(uint32_t)) {
            LOGE(LOG_ERROR, "oob_send failed");
            goto cleanup;
        }
        if (oob_send_s(sock, args->client->host_ptr+mem_offset, mem_this_thread) != mem_this_thread) {
            LOGE(LOG_ERROR, "oob_send failed");
            goto cleanup;
        }
    } else if (args->client->dir == MT_MEMCPY_DTOH) {
        if (oob_receive_s(sock, args->client->host_ptr+mem_offset, mem_this_thread) != mem_this_thread) {
            LOGE(LOG_ERROR, "oob_send failed");
            goto cleanup;
        }
    }

    if (close(sock) != 0) {
        LOGE(LOG_ERROR, "closing socket failed");
        goto cleanup;
    }
    ret = 0;
 cleanup:
    return (void*)ret;
}

int mt_memcpy_client(const char* server, uint16_t port, void* host_ptr, size_t size, enum mt_memcpy_direction dir, int thread_num)
{
    int ret = 1;
    struct addrinfo hints;
    struct addrinfo *addr = NULL;
    char port_str[6];
    if (sprintf(port_str, "%d", port) < 0) {
        printf("oob: sprintf failed.\n");
        return 1;
    }

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(server, port_str, &hints, &addr) != 0 || addr == NULL) {
        printf("error resolving hostname: %s\n", server);
        return 1;
    }

    struct client_args client = {
        .addr = addr,
        .host_ptr = host_ptr,
        .size = size,
        .dir = dir,
        .thread_num = thread_num
    };
    struct client_thread_args *copy_args;
    if ((copy_args = malloc(sizeof(struct client_thread_args) * thread_num)) == NULL) {
        return ret;
    }
    for (int i=0; i < thread_num; i++) {
        copy_args[i].client = &client;
        copy_args[i].thread_id = i;
        if ((ret = pthread_create(&copy_args[i].thread,
                                   NULL,
                                   mt_memcpy_client_thread,
                                   &copy_args[i])) != 0) {
            LOGE(LOG_ERROR, "mt_memcpy: failed to create client thread: %s", strerror(errno));
            goto cleanup;
        }
    }
    
    for (int i=0; i < thread_num; i++) {
        pthread_join(copy_args[i].thread, (void**)&ret);
        if (ret != 0) {
            LOGE(LOG_ERROR, "mt_memcpy: failed to copy memory.");
        }
    }
    ret = 0;
 cleanup:
    free(copy_args);
    freeaddrinfo(addr);
    return ret;
}
