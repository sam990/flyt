#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/msg.h>
#include <pthread.h>
#include <errno.h>

#include "msg-handler.h"
#include "log.h"
#include "cpu-utils.h"
#include "cpu-server-resource-controller.h"

#define CLIENTD_MQUEUE_PATH "/tmp/flyt-servernode-queue"
#define PROJ_ID 0x42

const char* SNODE_VIRTS_CHANGE_RESOURCES = "SNODE_VIRTS_CHANGE_RESOURCES";

pthread_t handler_thread;


static void resource_change_handler(const char* data, uint32_t *response) {
    splitted_str *split_str = split_string(data, ",");
    if (split_str->size != 2) {
        LOGE(LOG_ERROR, "Invalid message received from client manager: %s", data);
        *response = htonl(400);
        free_splitted_str(split_str);
        return;
    }

    uint32_t new_num_sm_cores = atoi(split_str->str[0]);
    uint64_t new_mem = strtoll(split_str->str[1], NULL, 10);

    // call change handler
    if (change_sm_cores(new_num_sm_cores) != 0) {
        *response = htonl(400);
        free_splitted_str(split_str);
        return;
    }

    if (set_mem_limit(new_mem) != 0) {
        *response = htonl(400);
        free_splitted_str(split_str);
        return;
    }

    *response = htonl(200);
    free_splitted_str(split_str);
    return;

}


static void *client_msg_handler(void *arg) {
    int clientd_mqueue_id = (int)((long)arg);
    struct msgbuf msg;

    uint64_t recv_type = msg_recv_id();
    uint64_t send_type = msg_send_id();

    while (1) {
        if (msgrcv(clientd_mqueue_id, &msg, sizeof(mqueue_msg), recv_type, 0) == -1) {
            LOGE(LOG_ERROR, "Error receiving message from client manager: %s", strerror(errno));
        }

        if (strncmp(msg.msg.cmd, SNODE_VIRTS_CHANGE_RESOURCES, strlen(SNODE_VIRTS_CHANGE_RESOURCES)) == 0) {
            LOGE(LOG_INFO, "Received message from client manager: %s", msg.msg.cmd);
            
            struct msgbuf_uint32 rsp;
            resource_change_handler(msg.msg.data, &rsp.data);
            rsp.mtype = send_type;
            if (msgsnd(clientd_mqueue_id, &rsp, sizeof(uint32_t), 0) == -1) {
                LOGE(LOG_ERROR, "Error sending response to client manager: %s", strerror(errno));
            }

        } else {
            LOGE(LOG_ERROR, "Unknown message received from client manager: %s", msg.msg.cmd);
        }
    }
    return NULL;
}


int init_listener(void)
{
    int clientd_mqueue_id;
    key_t key;

    if (access(CLIENTD_MQUEUE_PATH, F_OK) == -1) {
        if (mkdir(CLIENTD_MQUEUE_PATH, 0777) == -1) {
            LOGE(LOG_ERROR, "Error creating directory for client manager message queue: %s", strerror(errno));
            return -1;
        }
    }

    if ((key = ftok(CLIENTD_MQUEUE_PATH, PROJ_ID)) == -1) {
        LOGE(LOG_ERROR, "Error creating key for client manager message queue: %s", strerror(errno));
        return -1;
    }

    if ((clientd_mqueue_id = msgget(key, 0666 | IPC_CREAT)) == -1) {
        LOGE(LOG_ERROR, "Error creating client manager message queue: %s", strerror(errno));
        return -1;
    }

    pthread_create(&handler_thread, NULL, client_msg_handler, (void *)((long)clientd_mqueue_id));
}