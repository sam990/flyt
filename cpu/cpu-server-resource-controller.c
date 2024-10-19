
#include <pthread.h>
#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gsched.h"
#include "log.h"
#include "cpu-server-driver.h"
#include "cpu-server-resource-controller.h"
#include "device-management.h"
#include "cpu-server-client-mgr.h"

static int active_device = -1;
static uint64_t mem_limit = 0;
static uint64_t current_mm_usage = 0;

static volatile uint32_t new_num_sm_cores = 0;
static volatile uint64_t new_mem = 0;
static volatile int change_resource_flag = 0;

static CUcontext primaryCtx = NULL;
static CUcontext execCtx = NULL;

static int delete_context_resources() {
    // iterate over all client
    cricket_client_iter iter = get_client_iter();
    cricket_client *client;

    while ((client = get_next_client(&iter)) != NULL) {
        
        // delete all streams
        cudaStreamDestroy(client->default_stream);

        resource_map_iter *stream_iter = resource_map_init_iter(client->custom_streams);
        uint64_t idx;

        while ((idx = resource_map_iter_next(stream_iter)) != 0) {
            cudaStreamDestroy(client->custom_streams->list[idx].mapped_addr);
        }

        resource_map_free_iter(stream_iter);
        
    }

    return 0;

}


int change_sm_cores(uint32_t nm_sm_cores) {

    cudaDeviceSynchronize();

    CUcontext currentContext;
    CUresult res1;

    res1 = cuCtxGetCurrent(&currentContext);

    if (res1 != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res1, &errStr);
        LOGE(LOG_ERROR, "Failed to get current context: %s", errStr);
        return -1;
    }

    CUcontext newContext;
    CUresult res2;

    CUexecAffinityParam affinity_param;
    affinity_param.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity_param.param.smCount.val = nm_sm_cores;

    res2 = cuCtxCreate_v3(&newContext, &affinity_param, 1,  0, active_device);

    if (res2 != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res2, &errStr);
        LOGE(LOG_ERROR, "Failed to create new context: %s", errStr);
        return -1;
    }
    
    int ret = server_driver_ctx_state_restore(currentContext);

    if (ret != 0) {
        LOGE(LOG_DEBUG, "Unable to restore ctx state");
        return -1;
    }

    execCtx = newContext;

    LOGE(LOG_INFO, "Changed SM cores to %u", nm_sm_cores);

    
    cuCtxDestroy(currentContext);
    
    return 0;
    
}

void set_active_device(int device) {
    active_device = device;
}

int get_active_device() {
    return active_device;
}

int set_mem_limit(uint64_t limit) {
    mem_limit = limit;
    return 0;
}

uint64_t get_mem_limit() {
    return mem_limit;
}

int allow_mem_alloc(uint64_t size) {
    size_t gpu_memusage = get_gpu_memory_usage();
    LOGE(LOG_DEBUG, "Current memory usage: %zu, mem_limit: %llu", gpu_memusage, mem_limit);
    if (gpu_memusage + size > mem_limit) {
        return 0;
    }
    return 1;
} 

uint64_t get_mem_free() {
    return mem_limit - get_gpu_memory_usage();
}

void inc_mem_usage(uint64_t size) {
    current_mm_usage += size;
}

void dec_mem_usage(uint64_t size) {
    if (current_mm_usage >= size) {
        current_mm_usage -= size;
    }
    else {
        current_mm_usage = 0;
    }
}

void check_and_change_resource(void) {

    // LOGE(LOG_DEBUG, "Checking and changing resources: %d", change_resource_flag);

    if (change_resource_flag == 0) {
        return;
    }

    if (new_num_sm_cores != 0) {
        change_sm_cores(new_num_sm_cores);
        new_num_sm_cores = 0;
    }

    if (new_mem != 0) {
        set_mem_limit(new_mem);
        new_mem = 0;
    }

    change_resource_flag = 0;

}

int set_new_config(uint32_t __newsm, uint64_t __newmem) {
    struct cudaDeviceProp dev_prop;
    
    cudaError_t res = cudaGetDeviceProperties(&dev_prop, active_device);


    if (res != cudaSuccess) {
        LOGE(LOG_ERROR, "Failed to get device properties: %s", cudaGetErrorString(res));
        return -1;
    }

    if (__newsm == 0 ) {
        LOGE(LOG_ERROR, "Invalid number of SM cores: %u", __newsm);
        return -1;
    }

    if (__newmem == 0) {
        LOGE(LOG_ERROR, "Invalid memory limit: %" PRIu64, __newmem);
        return -1;
    }

    new_num_sm_cores = __newsm;
    new_mem = __newmem;

    LOGE(LOG_INFO, "Setting new configuration: %u SM cores, %" PRIu64 " memory limit", new_num_sm_cores, new_mem);

    change_resource_flag = 1;
    return 0;
}

int init_resource_controller(uint32_t num_sm_cores, uint64_t mem) {
    
    CUresult res = cuCtxGetCurrent(&primaryCtx);
    if (res != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        LOGE(LOG_ERROR, "Failed to get primary context: %s", errStr);
        return -1;
    }

    if (change_sm_cores(num_sm_cores) != 0) {
        LOGE(LOG_ERROR, "Failed to set sm cores configuration");
        return -1;
    }

    if (set_mem_limit(mem) != 0) {
        LOGE(LOG_ERROR, "Failed to set memory limit");
        return -1;
    }

    return init_device_management(active_device);
}

void set_primary_context() {
    cudaDeviceSynchronize();
    cuCtxPushCurrent(primaryCtx);
}

void unset_primary_context() {
    cuCtxPopCurrent(&primaryCtx);
}

void set_exec_context() {
    cudaDeviceSynchronize();
    cuCtxSetCurrent(execCtx);
}