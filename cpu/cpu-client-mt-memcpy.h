#ifndef _CPU_CLIENT_MT_MEMCPY_H_
#define _CPU_CLIENT_MT_MEMCPY_H_

#include <pthread.h>
#include <stdint.h>
#include "oob.h"

enum mt_memcpy_direction {
    MT_MEMCPY_HTOD,
    MT_MEMCPY_DTOH
};

int mt_memcpy_client(const char* server, uint16_t port, void* host_ptr, size_t size, enum mt_memcpy_direction dir, int thread_num);

#endif // _CPU_CLIENT_MT_MEMCPY_H_