#ifndef _CPU_SERVER_H_
#define _CPU_SERVER_H_

#include <stddef.h>

void cricket_main(size_t prog_num, size_t vers_num, uint32_t gpu_id, uint32_t num_sm_cores, size_t memory, uint32_t thread_mode);

#endif //_CPU_SERVER_H_
