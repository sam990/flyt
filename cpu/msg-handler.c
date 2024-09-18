/**
 * FILE: msg-handler.c
 * -------------------
 * Functions to generate message queue ids
 * for a process, across library source files.
 */

#include "msg-handler.h"
uint64_t msg_send_id() {
    return ((uint64_t)getpid()) << 32;
}

uint64_t msg_recv_id() {
    return getpid();
}
