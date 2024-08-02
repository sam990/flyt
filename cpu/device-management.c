#include <nvml.h>
#include <unistd.h>

#define MAX_PROCESS_COUNT 128

#include "log.h"

// Path: cpu/device-management.c

static int active_device = -1;
static nvmlDevice_t device = NULL;

static pid_t pid = -1;

int init_device_management(int device_id) {
    nvmlReturn_t result;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        LOGE(LOG_ERROR, "Failed to initialize NVML: %s", nvmlErrorString(result));
        return -1;
    }

    pid = getpid();
    active_device = device_id;

    result = nvmlDeviceGetHandleByIndex(active_device, &device);
    if (result != NVML_SUCCESS) {
        LOGE(LOG_ERROR, "Failed to get device handle: %s", nvmlErrorString(result));
        return -1;
    }
    return 0;
}


size_t get_gpu_memory_usage() {
    nvmlProcessInfo_t process_info[MAX_PROCESS_COUNT];
    unsigned int count = MAX_PROCESS_COUNT;

    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses(device, &count, process_info);
    if (result != NVML_SUCCESS) {
        LOGE(LOG_ERROR, "Failed to get process info: %s", nvmlErrorString(result));
        return 0;
    }

    for (unsigned int i = 0; i < count; i++) {
        LOGE(LOG_INFO, "process info pid %d usedGPUMemory %d: \n", process_info[i].pid, process_info[i].usedGpuMemory);
        if (process_info[i].pid == pid) {
            return process_info[i].usedGpuMemory;
        }
    }

    LOGE(LOG_ERROR, "Failed to find process info");
    return 0;
}
