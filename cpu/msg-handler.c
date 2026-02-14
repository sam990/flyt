/* Copyright (c) 2024-2026 SynerG Lab, IITB */

#include <stdio.h>
#include <stdlib.h>
#include "msg-handler.h"

uint64_t msg_send_id() {
    return ((uint64_t)getpid()) << 32;
}

uint64_t msg_recv_id() {
    return getpid();
}

uint64_t htonll(uint64_t value) {
    uint32_t high_part = htonl((uint32_t)(value >> 32)); // Get the high 32 bits and convert
    uint32_t low_part = htonl((uint32_t)(value & 0xFFFFFFFF)); // Get the low 32 bits and convert

    return ((uint64_t)low_part << 32) | high_part; // Recombine in network byte order
}

uint64_t ntohll(uint64_t value) {
    uint32_t high_part = ntohl((uint32_t)(value >> 32)); // Get the high 32 bits and convert
    uint32_t low_part = ntohl((uint32_t)(value & 0xFFFFFFFF)); // Get the low 32 bits and convert

    return ((uint64_t)low_part << 32) | high_part; // Recombine in host byte order
}
