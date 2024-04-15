
#include <pthread.h>
#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gsched.h"
#include "log.h"
#include "cpu-server-driver.h"

static int active_device = -1;
static uint64_t mem_limit = 0;
static uint64_t current_mm_usage = 0;

int change_sm_cores(uint32_t new_num_sm_cores) {
    GSCHED_EXCLUSIVE;

    cudaDeviceProp dev_prop;
    
    cudaError_t res = cudaGetDeviceProperties(&dev_prop, device_id);


    if (res != cudaSuccess) {
        LOGE(LOG_ERROR, "Failed to get device properties: %s", cudaGetErrorString(res));
        GSCHED_RELEASE;
        return -1;
    }

    if (new_num_sm_cores == 0 || new_num_sm_cores > dev_prop.multiProcessorCount) {
        LOGE(LOG_ERROR, "Invalid number of SM cores: %u", new_num_sm_cores);
        GSCHED_RELEASE;
        return -1;
    }

    CUcontext currentContext;
    CUcontext newContext;
    CUresult res;

    res = cuCtxGetCurrent(&currentContext);

    if (res != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "Failed to get current context: %s", cudaGetErrorString(res));
        GSCHED_RELEASE;
        return -1;
    }

    CUexecAffinityParam affinity;

    affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;

    affinity.param.smCount.val = smCounts[i];

    res = cuCtxCreate_v3(&newContext, affinity, 1, 0, active_device);

    if (res != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "Failed to create new context: %s", cudaGetErrorString(res));
        GSCHED_RELEASE;
        return -1;
    }

    cuCtxDestroy(currentContext);

    int ret = server_driver_elf_restore();

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to restore ELF: %d", ret);
        GSCHED_RELEASE;
        return -1;
    }

    ret = server_driver_function_restore();

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to restore functions: %d", ret);
        GSCHED_RELEASE;
        return -1;
    }

    GSCHED_RELEASE;

    return 0;
    
}

void set_active_device(int device) {
    GSCHED_EXCLUSIVE;
    active_device = device;
    GSCHED_RELEASE;
}

int get_active_device() {
    return active_device;
}

void set_mem_limit(uint64_t limit) {
    GSCHED_EXCLUSIVE;
    mem_limit = limit;
    GSCHED_RELEASE;
}

uint64_t get_mem_limit() {
    return mem_limit;
}

int allow_mem_alloc(uint64_t size) {
    return size <= mem_limit;
} 

void inc_mem_usage(uint64_t size) {
    current_mm_usage += size;
}

void dec_mem_usage(uint64_t size) {
    current_mm_usage -= size;
}