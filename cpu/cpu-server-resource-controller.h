#ifndef _CPU_SERVER_RESOURCE_CONTROLLER_H_
#define _CPU_SERVER_RESOURCE_CONTROLLER_H_

#include <stdint.h>

int change_sm_cores(uint32_t new_num_sm_cores);

void set_active_device(int device);

int get_active_device();

int set_mem_limit(uint64_t limit);

uint64_t get_mem_limit();

void inc_mem_usage(uint64_t size);

void dec_mem_usage(uint64_t size);

int allow_mem_alloc(uint64_t size);

void check_and_change_resource(void);

int set_new_config(uint32_t new_num_sm_cores, uint64_t new_mem);

int init_resource_controller(uint32_t num_sm_cores, uint64_t mem);

void set_primary_context();

void unset_primary_context();

void set_exec_context();

#define PRIMARY_CTX_RETAIN set_primary_context()
#define PRIMARY_CTX_RELEASE unset_primary_context()

#define SET_EXEC_CTX set_exec_context()

#endif // _CPU_SERVER_RESOURCE_CONTROLLER_H_