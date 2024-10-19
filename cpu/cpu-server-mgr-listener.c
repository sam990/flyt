#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/msg.h>
#include <pthread.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "msg-handler.h"
#include "log.h"
#include "cpu-utils.h"
#include "cpu-server-resource-controller.h"
#include "flyt-cr.h"
#include <signal.h>

#define SNODE_MQUEUE_PATH "/tmp/flyt-servernode-queue"
#define PROJ_ID 0x42

const char* SNODE_VIRTS_CHANGE_RESOURCES = "SNODE_VIRTS_CHANGE_RESOURCES";
const char* SNODE_VIRTS_CHECKPOINT = "SNODE_VIRTS_CHECKPOINT";
const char* SNODE_VIRTS_RESTORE = "SNODE_VIRTS_RESTORE";
const char* SNODE_VIRTS_DEALLOC = "SNODE_VIRTS_DEALLOC";

pthread_t handler_thread;

static volatile uint64_t recv_type = 0;
static volatile uint64_t send_type = 0;
static volatile int snode_mqueue_id = -1;


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

    *response = htonl(200);
    
    // This will only set the new configuration if the new configuration is valid
    // Actual change will be done when the next cuda call is made
    if (set_new_config(new_num_sm_cores, new_mem) != 0) {
        *response = htonl(400);
    }


    free_splitted_str(split_str);
}


static void *snode_msg_handler(void *arg) {

    struct msgbuf msg;

    while (1) {
        if (msgrcv(snode_mqueue_id, &msg, sizeof(mqueue_msg), recv_type, 0) == -1) {
            LOGE(LOG_ERROR, "Error receiving message from node manager: %s", strerror(errno));
        }

        if (strncmp(msg.msg.cmd, SNODE_VIRTS_CHANGE_RESOURCES, strlen(SNODE_VIRTS_CHANGE_RESOURCES)) == 0) {
            LOGE(LOG_INFO, "Received message from node manager: %s", msg.msg.cmd);
            
            struct msgbuf_uint32 rsp;
            resource_change_handler(msg.msg.data, &rsp.data);
            
            rsp.mtype = send_type;
            if (msgsnd(snode_mqueue_id, &rsp, sizeof(uint32_t), 0) == -1) {
                LOGE(LOG_ERROR, "Error sending response to node manager: %s", strerror(errno));
            }

        } 
        
        else if (strncmp(msg.msg.cmd, SNODE_VIRTS_CHECKPOINT, strlen(SNODE_VIRTS_CHECKPOINT)) == 0) {
            LOGE(LOG_INFO, "Received message from node manager: %s", msg.msg.cmd);
            
            struct msgbuf_uint32 rsp;
            
            int ret = flyt_create_checkpoint(msg.msg.data);

            rsp.mtype = send_type;
            rsp.data = ret == 0 ? htonl(200) : htonl(500);

            if (msgsnd(snode_mqueue_id, &rsp, sizeof(uint32_t), 0) == -1) {
                LOGE(LOG_ERROR, "Error sending response to node manager: %s", strerror(errno));
            }
        }

        else if (strncmp(msg.msg.cmd, SNODE_VIRTS_RESTORE, strlen(SNODE_VIRTS_RESTORE)) == 0) {
            LOGE(LOG_INFO, "Received message from node manager: %s", msg.msg.cmd);
            
            struct msgbuf_uint32 rsp;
            
            int ret = flyt_restore_checkpoint(msg.msg.data);

            rsp.mtype = send_type;
            rsp.data = ret == 0 ? htonl(200) : htonl(500);

            if (msgsnd(snode_mqueue_id, &rsp, sizeof(uint32_t), 0) == -1) {
                LOGE(LOG_ERROR, "Error sending response to node manager: %s", strerror(errno));
            }
        }

        else if (strncmp(msg.msg.cmd, SNODE_VIRTS_DEALLOC, strlen(SNODE_VIRTS_DEALLOC)) == 0) {
            LOGE(LOG_INFO, "Received message from node manager: %s", msg.msg.cmd);
            
            struct msgbuf_uint32 rsp;
            
            int ret = dealloc_client_resources();

            rsp.mtype = send_type;
            rsp.data = ret == 0 ? htonl(200) : htonl(500);

            if (msgsnd(snode_mqueue_id, &rsp, sizeof(uint32_t), 0) == -1) {
                LOGE(LOG_ERROR, "Error sending response to node manager: %s", strerror(errno));
            }
        }
        
        else {
            LOGE(LOG_ERROR, "Unknown message received from node manager: %s", msg.msg.cmd);
        }
    }
    return NULL;
}

void send_initialised_msg() {
    struct msgbuf_uint32 msg;
    msg.mtype = send_type;
    msg.data = htonl(200);
    printf("mqueue_id = %d type %ld data %u\n", snode_mqueue_id, msg.mtype, msg.data);
    if (msgsnd(snode_mqueue_id, &msg, sizeof(msg.data), 0) == -1) {
        LOGE(LOG_ERROR, "Error sending initialisation message to node manager: %s", strerror(errno));
    }
}

int init_listener(int rpc_id)
{
    recv_type = (uint64_t)rpc_id;
    send_type = ((uint64_t)rpc_id) << 32;
    key_t key;

    printf("send_type = %lu\n", send_type);
    if (access(SNODE_MQUEUE_PATH, F_OK) == -1) {
        if (mkdir(SNODE_MQUEUE_PATH, 0777) == -1) {
            LOGE(LOG_ERROR, "Error creating directory for node manager message queue: %s", strerror(errno));
            return -1;
        }
    }

    if ((key = ftok(SNODE_MQUEUE_PATH, PROJ_ID)) == -1) {
        LOGE(LOG_ERROR, "Error creating key for node manager message queue: %s", strerror(errno));
        return -1;
    }

    if ((snode_mqueue_id = msgget(key, 0666 | IPC_CREAT)) == -1) {
        LOGE(LOG_ERROR, "Error creating node manager message queue: %s", strerror(errno));
        return -1;
    }

    pthread_create(&handler_thread, NULL, snode_msg_handler, NULL);
}
