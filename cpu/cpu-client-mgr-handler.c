#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/msg.h>
#include <arpa/inet.h>
#include <errno.h>

#include "cpu-common.h"
#include "cpu-client-mgr-handler.h"
#include "log.h"

#define CLIENTD_MQUEUE_PATH "/tmp/flyt-client-mgr"
#define PROJ_ID 0x42

const char* CLIENTD_VCUDA_PAUSE = "CLIENTD_VCUDA_PAUSE";
const char* CLIENTD_VCUDA_CHANGE_VIRT_SERVER = "CLIENTD_VCUDA_CHANGE_VIRT_SERVER";
const char* CLIENTD_VCUDA_RESUME = "CLIENTD_VCUDA_RESUME";
const char* PING = "PING";


typedef struct mqueue_msg {
    char cmd[64];
    char data[64];
} mqueue_msg;

struct msgbuf {
    long mtype;       /* message type, must be > 0 */
    mqueue_msg msg;    /* message data */
};

static pthread_t handler_thread;
static volatile uint8_t keep_handler_alive = 1;


static void* cpu_client_mgr_handler(void* arg) {
    
    uint64_t recv_type = getpid();
    uint64_t send_type = recv_type << 32;
    
    int clientd_mqueue_id = (int)arg;

    struct msgbuf msg;

    while (keep_handler_alive) {

        if (msgrcv(clientd_mqueue_id, &msg, sizeof(mqueue_msg), recv_type, 0) == -1) {
            LOGE(LOG_ERROR, "Error receiving message from client manager: %s", strerror(errno));
        }

       if (strncmp(msg.msg.cmd, PING, 64) == 0) {
            strcpy(msg.msg.cmd, "200");
            strcpy(msg.msg.data, "PONG");
            msg.mtype = send_type;
            msgsnd(clientd_mqueue_id, &msg, sizeof(mqueue_msg), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_PAUSE, 64) == 0) {
            sem_wait(&access_sem);
            strcpy(msg.msg.cmd, "200");
            strcpy(msg.msg.data, "VCUDA PAUSED");
            msg.mtype = send_type;
            msgsnd(clientd_mqueue_id, &msg, sizeof(mqueue_msg), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_RESUME, 64) == 0) {
            sem_post(&access_sem);
            strcpy(msg.msg.cmd, "200");
            strcpy(msg.msg.data, "VCUDA RESUMED");
            msg.mtype = send_type;
            msgsnd(clientd_mqueue_id, &msg, sizeof(mqueue_msg), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_CHANGE_VIRT_SERVER, 64) == 0) {
            char *server_info = strdup(msg.msg.data);
            
            change_server(server_info);

            strcpy(msg.msg.cmd, "200");
            strcpy(msg.msg.data, "VCUDA VIRT SERVER CHANGED");
            msg.mtype = send_type;
            msgsnd(clientd_mqueue_id, &msg, sizeof(mqueue_msg), 0);
        }
        else {
            LOGE(LOG_ERROR, "Unknown command received from client manager: %s", msg.msg.cmd);
        }

    }

    return NULL;
}

char* get_virt_server_info(int mqueue_id, uint64_t recv_type) {
    struct msgbuf msg;
    int read = msgrecv(mqueue_id, &msg, sizeof(mqueue_msg), recv_type, 0);

    if (read == -1) {
        LOGE(LOG_ERROR, "Error receiving message from client manager: %s", strerror(errno));
        return NULL;
    }

    if (strncmp(msg.msg.cmd, "200", 64) == 0) {
        return strdup(msg.msg.data);
    }

    return NULL;
}


void init_handler_thread(int clientd_mqueue_id) {
    handler_thread = (pthread_t*)malloc(sizeof(pthread_t));
    pthread_create(&handler_thread, NULL, cpu_client_mgr_handler, (void*)clientd_mqueue_id);
}

void stop_client_mgr() {
    keep_handler_alive = 0;
    pthread_join(handler_thread, NULL);
}

char* init_client_mgr() {
    key_t key = ftok(CLIENTD_MQUEUE_PATH, PROJ_ID);
    if (key == -1) {
        perror("ftok");
        exit(EXIT_FAILURE);
    }

    int clientd_mqueue_id = msgget(key, IPC_CREAT | 0666);
    if (clientd_mqueue_id == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }


    pid_t pid = getpid();

    uint64_t recv_id = pid;
    uint64_t send_id = recv_id << 32;

    uint64_t recv_id = getpid();

    uint32_t npid = htonl(pid);

    msgsnd(clientd_mqueue_id, &npid, sizeof(uint32_t), 1);

    char *virt_server_info = get_virt_server_info(clientd_mqueue_id, recv_id);

    if (virt_server_info == NULL) {
        return NULL;
    }

    init_handler_thread(clientd_mqueue_id);

    return virt_server_info;
}