/* Copyright (c) 2024-2026 SynerG Lab, IITB */

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/msg.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "cpu-common.h"
#include "cpu-client-mgr-handler.h"
#include "log.h"
#include "msg-handler.h"

#define CLIENTD_MQUEUE_PATH "/tmp/flyt-client-mgr"

const char* CLIENTD_VCUDA_PAUSE = "CLIENTD_VCUDA_PAUSE";
const char* CLIENTD_VCUDA_CHANGE_VIRT_SERVER = "CLIENTD_VCUDA_CHANGE_VIRT_SERVER";
const char* CLIENTD_VCUDA_RESUME = "CLIENTD_VCUDA_RESUME";
const char* CLIENTD_RMGR_CONNECT = "CLIENTD_RMGR_CONNECT";
const char* CLIENTD_RMGR_DISCONNECT = "CLIENTD_RMGR_DISCONNECT";
const char* PING = "PING";



static pthread_t handler_thread;
static volatile uint8_t keep_handler_alive = 1;


static void* cpu_client_mgr_handler(void* arg) {
    
    uint64_t recv_type = msg_recv_id();
    uint64_t send_type = msg_send_id();
    
    int clientd_mqueue_id = (int)((long)arg);

    struct msgbuf msg;

    while (keep_handler_alive) {

        if (msgrcv(clientd_mqueue_id, &msg, sizeof(mqueue_msg), recv_type, 0) == -1) {
            LOGE(LOG_ERROR, "Error receiving message from client manager: %s", strerror(errno));
        }

       if (strncmp(msg.msg.cmd, PING, sizeof(msg.msg.cmd)) == 0) {
            struct msgbuf_uint32 resp;
            resp.mtype = send_type;
            resp.data = htonl(200);
            msgsnd(clientd_mqueue_id, &resp, sizeof(resp.data), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_PAUSE, sizeof(msg.msg.cmd)) == 0) {
            LOGE(LOG_INFO, "received VCUDA_PAUSE cmd:before sem_lock");
            pthread_rwlock_wrlock(&access_sem);
            struct msgbuf_uint32 resp;
            resp.mtype = send_type;
            resp.data = htonl(200);
            LOGE(LOG_INFO, "received VCUDA_PAUSE cmd:after sem_lock");
            msgsnd(clientd_mqueue_id, &resp, sizeof(resp.data), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_RESUME, sizeof(msg.msg.cmd)) == 0) {
            resume_connection();
            pthread_rwlock_unlock(&access_sem);
            struct msgbuf_uint32 resp;
            resp.mtype = send_type;
            resp.data = htonl(200);
            msgsnd(clientd_mqueue_id, &resp, sizeof(resp.data), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_CHANGE_VIRT_SERVER, sizeof(msg.msg.cmd)) == 0) {
            char *server_info = strdup(msg.msg.data);
            
            change_server(server_info);

            struct msgbuf_uint32 resp;
            resp.mtype = send_type;
            resp.data = htonl(200);
            msgsnd(clientd_mqueue_id, &resp, sizeof(resp.data), 0);
        }
        else {
            LOGE(LOG_ERROR, "Unknown command received from client manager: %s", msg.msg.cmd);
        }

    }

    return NULL;
}

char* get_virt_server_info(int mqueue_id, uint64_t recv_type) {
    LOGE(LOG_DEBUG, "Receiving server info message from client manager");
    struct msgbuf msg;
    int read = msgrcv(mqueue_id, &msg, sizeof(mqueue_msg), recv_type, 0);
    LOGE(LOG_DEBUG, "Got server info message from client manager");

    if (read == -1) {
        LOGE(LOG_ERROR, "Error receiving message from client manager: %s", strerror(errno));
        return NULL;
    }

    if (strncmp(msg.msg.cmd, "200", sizeof(msg.msg.cmd)) == 0) {
        return strdup(msg.msg.data);
    }

    return NULL;
}

void init_handler_thread(int clientd_mqueue_id) {
    pthread_create(&handler_thread, NULL, cpu_client_mgr_handler, (void*)((long)clientd_mqueue_id));
}

void stop_client_mgr() {
    keep_handler_alive = 0;
    pthread_join(handler_thread, NULL);
}

int get_sm_core_value() {
    const char *sm_core_str = getenv("SM_CORE");
    if (sm_core_str) {
        char *endptr;
        long value = strtol(sm_core_str, &endptr, 10);

        // Check if the entire string was converted to a number
        if (*endptr == '\0' && errno != ERANGE) {
            return (int)value;  // Return the valid integer value
        } else {
            // Handle error if the string was not a valid integer
            fprintf(stderr, "Invalid value for SM_CORE: %s\n", sm_core_str);
            return -1;
        }
    }
    return -1;  // Return -1 if the environment variable is not set
}

char* init_client_mgr() {
    LOGE(LOG_DEBUG, "My Connecting to client manager");
    if (access(CLIENTD_MQUEUE_PATH, F_OK) == -1) {
        mknod(CLIENTD_MQUEUE_PATH, S_IFREG | 0666, 0);
    }

    key_t key = ftok(CLIENTD_MQUEUE_PATH, PROJ_ID);
    if (key == -1) {
        perror("ftok");
        exit(EXIT_FAILURE);
    }

    LOGE(LOG_DEBUG, "msgqueue key: %d\n", key);

    int clientd_mqueue_id = msgget(key, IPC_CREAT | 0666);
    if (clientd_mqueue_id == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }

    uint64_t recv_id = msg_recv_id();
    uint64_t send_id = msg_send_id();
    int sm_core = get_sm_core_value();

    struct msgbuf msg;
    msg.mtype = 1;
    strncpy(msg.msg.cmd, CLIENTD_RMGR_CONNECT, sizeof(msg.msg.cmd));
    sprintf(msg.msg.data, "%d,%d", recv_id,sm_core);

    msgsnd(clientd_mqueue_id, &msg, sizeof(msg.msg), 0);

    char *virt_server_info = get_virt_server_info(clientd_mqueue_id, recv_id);
    
    if (virt_server_info == NULL) {
        return NULL;
    }

    init_handler_thread(clientd_mqueue_id);

    return virt_server_info;
}

void deinit_client_mgr() {
    LOGE(LOG_DEBUG, "Disconnecting client manager");
    if (access(CLIENTD_MQUEUE_PATH, F_OK) == -1) {
        mknod(CLIENTD_MQUEUE_PATH, S_IFREG | 0666, 0);
    }

    key_t key = ftok(CLIENTD_MQUEUE_PATH, PROJ_ID);
    if (key == -1) {
        LOGE(LOG_ERROR, "ftok ");
	return;
    }

    LOGE(LOG_DEBUG, "msgqueue key: %d", key);

    int clientd_mqueue_id = msgget(key, IPC_CREAT | 0666);
    if (clientd_mqueue_id == -1) {
        LOGE(LOG_ERROR, "msgget ");
	return;
    }

    uint64_t recv_id = msg_recv_id();
    uint64_t send_id = msg_send_id();

    struct msgbuf msg;
    msg.mtype = 1;
    strncpy(msg.msg.cmd, CLIENTD_RMGR_DISCONNECT, sizeof(msg.msg.cmd));
    sprintf(msg.msg.data, "%d", recv_id);

    LOGE(LOG_DEBUG, "Sending Disconnect command to client manager");
    msgsnd(clientd_mqueue_id, &msg, sizeof(msg.msg), 0);

    struct msgbuf rsp;
    int read = msgrcv(clientd_mqueue_id, &rsp, sizeof(mqueue_msg), 2, 0);
    LOGE(LOG_DEBUG, "Got disconnect response from client manager");

    if (read == -1) {
        LOGE(LOG_ERROR, "Error receiving message from client manager: %s", strerror(errno));
        return;
    }
    else if (strncmp("200", msg.msg.cmd, sizeof("200")) >= 0) {
        LOGE(LOG_ERROR, "received wrong response command for disconnect request : %s", msg.msg.cmd);
        return;
    }
    LOGE(LOG_INFO, "completed deinit request and response");
    
}
