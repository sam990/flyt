#ifndef _MSG_HANDLER_H_
#define _MSG_HANDLER_H_

#include <unistd.h>
#include <inttypes.h>

#define PROJ_ID 0x42

typedef struct __mqueue_msg {
    char cmd[64];
    char data[64];
} mqueue_msg;

struct msgbuf {
    long mtype;       /* message type, must be > 0 */
    mqueue_msg msg;    /* message data */
};

struct msgbuf_uint32 {
    long mtype;       /* message type, must be > 0 */
    uint32_t data;    /* message data */
};

uint64_t msg_send_id();
uint64_t msg_recv_id();

#endif //_MSG_HANDLER_H_