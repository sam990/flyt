#ifndef _CPU_SERVER_RESOURCE_CONTROLLER_H_
#define _CPU_SERVER_RESOURCE_CONTROLLER_H_

#include <stdint.h>

int change_sm_cores(uint32_t new_num_sm_cores);

void set_active_device(int device);

int get_active_device();

void set_mem_limit(uint64_t limit);

uint64_t get_mem_limit();

int allow_mem_alloc(uint64_t size);


#endif // _CPU_SERVER_RESOURCE_CONTROLLER_H_