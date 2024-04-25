#ifndef __FLYT_DEVICE_MANAGEMENT_H__
#define __FLYT_DEVICE_MANAGEMENT_H__

#include <stddef.h>

int init_device_management(int device_id);
size_t get_gpu_memory_usage();


#endif // __FLYT_DEVICE_MANAGEMENT_H__