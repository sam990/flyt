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
#include "cpu-utils.h"
#include <assert.h> 

#define CLIENTD_MQUEUE_PATH "/tmp/flyt-client-mgr"

const char* CLIENTD_VCUDA_PAUSE = "CLIENTD_VCUDA_PAUSE";
const char* CLIENTD_VCUDA_CHANGE_VIRT_SERVER = "CLIENTD_VCUDA_CHANGE_VIRT_SERVER";
const char* CLIENTD_VCUDA_RESUME = "CLIENTD_VCUDA_RESUME";
const char* PING = "PING";



static pthread_t handler_thread;
static volatile uint8_t keep_handler_alive = 1;


static void* cpu_client_mgr_handler(void* arg) {
    // same mq, same ids as the virst-server-req sender,
    // but response dosnt get handled in this (wrong)
    // thread.
    uint64_t recv_type = msg_recv_id();
    uint64_t send_type = msg_send_id();
    
    int clientd_mqueue_id = (int)((long)arg);

    struct msgbuf msg;

    while (keep_handler_alive) {

        if (msgrcv(clientd_mqueue_id, &msg, sizeof(mqueue_msg), recv_type, 0) == -1) {
            LOGE(LOG_ERROR, "Error receiving message from client manager: %s", strerror(errno));
        }
        LOGE(LOG_DEBUG, "Got a message on the library handler thread, non-ivshmem.\n");
       if (strncmp(msg.msg.cmd, PING, 64) == 0) {
            struct msgbuf_uint32 resp;
            resp.mtype = send_type;
            resp.data = htonl(200);
            msgsnd(clientd_mqueue_id, &resp, sizeof(resp.data), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_PAUSE, 64) == 0) {
            pthread_rwlock_wrlock(&access_sem); // pause. no more reqs.
            struct msgbuf_uint32 resp;
            resp.mtype = send_type;
            resp.data = htonl(200);
            msgsnd(clientd_mqueue_id, &resp, sizeof(resp.data), 0);
        }
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_RESUME, 64) == 0) {
            resume_connection();
            pthread_rwlock_unlock(&access_sem);
            struct msgbuf_uint32 resp;
            resp.mtype = send_type;
            resp.data = htonl(200);
            msgsnd(clientd_mqueue_id, &resp, sizeof(resp.data), 0);
        }
        // happens before client resume
        // creates a new rpc connection with the new virt server instance.
        else if (strncmp(msg.msg.cmd, CLIENTD_VCUDA_CHANGE_VIRT_SERVER, 64) == 0) {
            char *server_str = strdup(msg.msg.data);
            // will fail.
            // in migration path, server_str sent by client_mgr is just an IP address.
            server_info_t *server_info = parse_server_str(server_str);
            
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

    if (strncmp(msg.msg.cmd, "200", 64) == 0) {
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

server_info_t *parse_server_str(char *server_str) {
    server_info_t *res = malloc(sizeof(server_info_t));
    assert(res);

    // split and assign.
    splitted_str *splitted = split_string(server_str, ",");
    
    if (splitted == NULL) {
        LOGE(LOG_ERROR, "error splitting server info: %s", server_str);
        exit(1);
    }
    if (splitted->size != 4) {
        LOGE(LOG_ERROR, "error parsing server info: %s", server_str);
        exit(1);
    }

    strcpy(res->server_ip, splitted->str[0]);
    printf("server ip from cmgr: %s\n", res->server_ip);

    res->rpc_id = strtoul(splitted->str[1], NULL, 10);
    printf("rpc_id from cmgr: %d\n", res->rpc_id);

    res->shm_enable = strtoul(splitted->str[2], NULL, 10);
    printf("shm_enable from cmgr: %d\n", res->shm_enable);

    strcpy(res->shm_backend, splitted->str[3]);
    printf("be from cmgr: %s\n", res->shm_backend);

    free_splitted_str(splitted);
    splitted = NULL;

    return res;
}

server_info_t *init_client_mgr() {
    LOGE(LOG_DEBUG, "Connecting to client manager");
    if (access(CLIENTD_MQUEUE_PATH, F_OK) == -1) {
        mknod(CLIENTD_MQUEUE_PATH, S_IFREG | 0666, 0);
    }

    // this key value is the same as the one used on 
    // the manager process.
    key_t key = ftok(CLIENTD_MQUEUE_PATH, PROJ_ID);
    if (key == -1) {
        perror("ftok");
        exit(EXIT_FAILURE);
    }

    LOGE(LOG_DEBUG, "msgqueue key: %d\n", key);

    // _id is a process-local handle to the message queue associated with the
    // given key. This creates the message queue, "connected"
    // to another process that initiated an MQ with the same key
    //
    // This is the library handle to the client manager daemon.
    // Needs to be global.
    int clientd_mqueue_id = msgget(key, IPC_CREAT | 0666);
    if (clientd_mqueue_id == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }

    pid_t pid = getpid();

    // send ID: the recvID of a the messageQ endpoint
    // that this process is sending to. 
    // ---
    // recv_id: The id that other mQ enpoints must send to
    // This enables the message ueue backend to have
    // multiple processes reading and writing.
    uint64_t recv_id = msg_recv_id(); // == getpid(), different per process
    uint64_t send_id = msg_send_id(); // different per process

    struct msgbuf_uint32 msg;
    msg.mtype = 1;
    msg.data = htonl(pid);

    // send pid to cluster manager.
    msgsnd(clientd_mqueue_id, &msg, sizeof(msg.data), 0);

    // wait for serverIP from clustermgr.
    // also get shm_enabled, shm_be_path
    // via strdup.
    char *virt_server_info = get_virt_server_info(clientd_mqueue_id, recv_id);
    printf("From Control Plane: %s#\n", virt_server_info);
    printf("sup\n");
    
    if (virt_server_info == NULL) {
        return NULL;
    }

    server_info_t *virt_server = parse_server_str(virt_server_info);
    virt_server->clientd_mqueue_id = clientd_mqueue_id;
    printf("unix mq id from lib: %d\n", virt_server->clientd_mqueue_id);

    init_handler_thread(clientd_mqueue_id); 

    // test send-recv here, post thread-creation.
    // haha once a message queue id is created, 
    // only one thread should rec the messages.
    // (recv_id, send_id) is unique.
    // send only needs to specify what queue to use
    // - can use common queue.
   // msgsnd(clientd_mqueue_id, &msg, sizeof(msg.data), 0);

    return virt_server;
}
