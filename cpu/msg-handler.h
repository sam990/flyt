#ifndef _MSG_HANDLER_H_
#define _MSG_HANDLER_H_

#include <unistd.h>
#include <inttypes.h>
#include <arpa/inet.h>

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

struct metricsInfo {
    uint64_t totalThreadResource;
    uint64_t kernelElapseTimeMS;
    uint64_t kernelLaunchCount;

    uint64_t memUsage;

    uint64_t totalMemcpySize;
    uint64_t memcpyElapseTimeMS;
    uint64_t memcpyCount;

    uint64_t totalStreams;
    uint64_t totalEvents;
};

struct msgbuf_metrics {
    long mtype;       /* message type, must be > 0 */
    struct metricsInfo msg;
};

uint64_t msg_send_id();

uint64_t msg_recv_id();

uint64_t htonll(uint64_t value);

uint64_t ntohll(uint64_t value);

#endif //_MSG_HANDLER_H_
