// profiler.h
// CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>

// CUPTI headers
#include <cupti_callbacks.h>
#include <cupti_profiler_target.h>
#include <cupti_driver_cbid.h>
#include <cupti_target.h>
#include <cupti_activity.h>

#ifdef __cplusplus
extern "C" {
#endif

void ProfilerCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    const void *pCallbackData);

int InitializeInjection();

int getMetrics(uint64_t *array, int size);

#ifdef __cplusplus
}
#endif

