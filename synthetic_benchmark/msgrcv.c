#include <stdio.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>


#define CLIENTD_MQUEUE_PATH "/tmp/flyt-client-mgr"
#define PROJ_ID 0x42

typedef struct __mqueue_msg {
	    char cmd[64];
	        char data[64];
} mqueue_msg;

struct msgbuf {
	    long mtype;       /* message type, must be > 0 */
	        mqueue_msg msg;    /* message data */
};

int main() {
	
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

	    struct msgbuf msg;
	  msgrcv(clientd_mqueue_id, &msg, sizeof(mqueue_msg), 1, 0);
	printf("received message\n");
	return 0;
}	
