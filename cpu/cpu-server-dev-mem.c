#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <math.h>

#include <cuda_runtime.h>
#include "cpu-utils.h"
#include "log.h"

static size_t granularity;
static CUmemAllocationProp prop = {};
static CUmemAccessDesc accessDesc = {};
static size_t pitch_width;

static size_t cudaCalculateMallocPitch(size_t width){
  void *p;
  size_t pitch;
  cudaMallocPitch(&p, &pitch, width, 2);
  cudaFree(p);
  return pitch;
}

int init_server_dev_mem(int device_id) {
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    CUresult res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    if (res != CUDA_SUCCESS) {
        LOGE(LOG_WARNING, "%s: cuMemGetAllocationGranularity error: %d", __FUNCTION__, res);
        return -1;
    }

    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device_id;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    LOGE(LOG_DEBUG, "Granularity: %zu", granularity);

    return 0;
}

cudaError_t dev_mem_alloc(void **dev_ptr, size_t size, int va_specified, size_t *padded_size_out) {

    CUresult res;

    CUmemGenericAllocationHandle allocHandle;
    
    size_t padded_size = (((size - 1) / granularity) + 1) * granularity;

    CUdeviceptr dptr;

    CUdeviceptr req_addr = *((CUdeviceptr *)dev_ptr);

    if (va_specified) {
        res = cuMemAddressReserve(&dptr, padded_size, 0, req_addr, 0);
    } else {
        res = cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
    }

    if (res != CUDA_SUCCESS) {
        LOGE(LOG_WARNING, "%s: cuMemAddressReserve error: %d", __FUNCTION__, res);
        return cudaErrorMemoryAllocation;
    }


    if (va_specified && (dptr != req_addr)) {
        LOGE(LOG_WARNING, "%s: Required address cannnot be reserved: %p", __FUNCTION__, req_addr);
        return cudaErrorMemoryAllocation;
    }
    
    res = cuMemCreate(&allocHandle, padded_size, &prop, 0); 

    if (res != CUDA_SUCCESS) {
        LOGE(LOG_WARNING, "%s: cuMemCreate error: %d", __FUNCTION__, res);
        return cudaErrorMemoryAllocation;
    }

    res = cuMemMap(dptr, padded_size, 0, allocHandle, 0);

    if (res != CUDA_SUCCESS) {
        LOGE(LOG_WARNING, "%s: cuMemMap error: %d", __FUNCTION__, res);
        cuMemRelease(allocHandle);
        return cudaErrorMemoryAllocation;
    }

    res = cuMemRelease(allocHandle);
    if (res != CUDA_SUCCESS) {
        LOGE(LOG_WARNING, "%s: cuMemRelease error: %d", __FUNCTION__, res);
        cuMemUnmap(dptr, padded_size);
        cuMemAddressFree(dptr, padded_size);
        return cudaErrorMemoryAllocation;
    }

    res = cuMemSetAccess(dptr, padded_size, &accessDesc, 1);
    if (res != CUDA_SUCCESS) {
        LOGE(LOG_WARNING, "%s: cuMemSetAccess error: %d", __FUNCTION__, res);
        cuMemUnmap(dptr, padded_size);
        cuMemAddressFree(dptr, padded_size);
        return cudaErrorMemoryAllocation;
    }

    *((CUdeviceptr *)dev_ptr) = dptr;
    *padded_size_out = padded_size;

    


    return cudaSuccess;

}

cudaError_t dev_mem_alloc_3d(void **dev_ptr, size_t depth, size_t height, size_t width, size_t *pitch_out, size_t *padded_size_out) {
    *pitch_out = cudaCalculateMallocPitch(width);
    return dev_mem_alloc(dev_ptr, *pitch_out * height * depth, 0, padded_size_out);
}

cudaError_t dev_mem_alloc_pitched(void **dev_ptr, size_t width, size_t height, size_t *pitch_out, size_t *padded_size_out) {
    *pitch_out = cudaCalculateMallocPitch(width);
    return dev_mem_alloc(dev_ptr, *pitch_out * height, 0, padded_size_out);
}



cudaError_t free_dev_mem(void *ptr, size_t padded_size) {
    if (ptr == NULL)
        return cudaSuccess;
    if (cuMemUnmap((CUdeviceptr)ptr, padded_size) != CUDA_SUCCESS || cuMemAddressFree((CUdeviceptr)ptr, padded_size) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    return cudaSuccess;
}

