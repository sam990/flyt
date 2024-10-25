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
 
typedef struct __mqueue_msg {
    char cmd[64];
    char data[64];
} mqueue_msg;
 
struct msgbuf {
    long mtype;       /* message type, must be > 0 */
    mqueue_msg msg;    /* message data */
};
 
 
#define PROJ_ID 0x42
#define CLIENTD_MQUEUE_PATH "/tmp/flyt-client-mgr"
 
int main() {
 
    if (access(CLIENTD_MQUEUE_PATH, F_OK) == -1) {
        mknod(CLIENTD_MQUEUE_PATH, S_IFREG | 0666, 0);
    }
 
    key_t key = ftok(CLIENTD_MQUEUE_PATH, PROJ_ID);
    if (key == -1) {
        perror("ftok");
        exit(EXIT_FAILURE);
    }
 
    printf("message queue key: %d\n", key);
 
    int clientd_mqueue_id = msgget(key, IPC_CREAT | 0666);
    if (clientd_mqueue_id == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }
 
    struct msgbuf msg;
 
    while(msgrcv(clientd_mqueue_id, &msg, sizeof(mqueue_msg), 0, 0) != -1) {
        printf("received message type: %lu\n", msg.mtype);
        printf("received message command: %s\n", msg.msg.cmd);
    }
 
    return 0;
}
