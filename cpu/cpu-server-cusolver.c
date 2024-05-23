#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

//for strerror
#include <string.h>
#include <errno.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#define WITH_RECORDER
#include "api-recorder.h"
#include "cpu-server-cusolver.h"
#include "cpu-server-client-mgr.h"
#include "gsched.h"



int cusolver_init(int bypass, resource_mg *streams, resource_mg *memory)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cusolver, bypass);
    return ret;
}

resource_mg *cusolver_get_rm(void)
{
    return &rm_cusolver;
}

int cusolver_deinit(void)
{
    resource_mg_free(&rm_cusolver);
    return 0;

}

bool_t rpc_cusolverdncreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "cusolverDnCreate");

    result->err = cusolverDnCreate((cusolverDnHandle_t*)&result->ptr_result_u.ptr);
    RECORD_RESULT(ptr_result_u, *result);
    resource_mg_create(&rm_cusolver, (void*)result->ptr_result_u.ptr);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cusolverdnsetstream_1_svc(ptr handle, ptr stream, int *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    RECORD_API(rpc_cusolverdnsetstream_1_argument);
    RECORD_ARG(1, handle);
    RECORD_ARG(2, stream);
    LOGE(LOG_DEBUG, "cusolverDnSetStream");

    GET_CLIENT(*result)

    void *stream_ptr;
    GET_STREAM(stream_ptr, stream, *result);

    *result = cusolverDnSetStream(resource_mg_get_or_null(&rm_cusolver, (void*)handle),
                                  (cudaStream_t)stream_ptr);
    RECORD_RESULT(integer, *result);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cusolverdndgetrf_buffersize_1_svc(ptr handle, int m, int n, ptr A, int lda, int_result *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    LOGE(LOG_DEBUG, "cusolverDnDgetrf_buffersize");

    GET_CLIENT(result->err);
    void *mem_ptr_A;
    GET_MEMORY(mem_ptr_A, A, result->err);


    result->err = cusolverDnDgetrf_bufferSize(resource_mg_get_or_null(&rm_cusolver, (void*)handle),
                                              m, n,
                                              mem_ptr_A,
                                              lda, &result->int_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cusolverdndgetrf_1_svc(ptr handle, int m, int n, ptr A, int lda, ptr Workspace, ptr devIpiv, ptr devInfo, int *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    LOGE(LOG_DEBUG, "cusolverDnDgetrf");

    GET_CLIENT(*result);
    void *mem_ptr_A, *mem_ptr_Workspace, *mem_ptr_devIpiv, *mem_ptr_devInfo;
    GET_MEMORY(mem_ptr_A, A, *result);
    GET_MEMORY(mem_ptr_Workspace, Workspace, *result);
    GET_MEMORY(mem_ptr_devIpiv, devIpiv, *result);
    GET_MEMORY(mem_ptr_devInfo, devInfo, *result);

    *result = cusolverDnDgetrf(resource_mg_get_or_null(&rm_cusolver, (void*)handle),
                               m, n,
                               mem_ptr_A,
                               lda,
                               mem_ptr_Workspace,
                               mem_ptr_devIpiv,
                               mem_ptr_devInfo);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cusolverdndgetrs_1_svc(ptr handle, int trans, int n, int nrhs, ptr A, int lda, ptr devIpiv, ptr B, int ldb, ptr devInfo, int *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    LOGE(LOG_DEBUG, "cusolverDnDgetrs");

    GET_CLIENT(*result);
    void *mem_ptr_A, *mem_ptr_devIpiv, *mem_ptr_B, *mem_ptr_devInfo;
    GET_MEMORY(mem_ptr_A, A, *result);
    GET_MEMORY(mem_ptr_devIpiv, devIpiv, *result);
    GET_MEMORY(mem_ptr_B, B, *result);
    GET_MEMORY(mem_ptr_devInfo, devInfo, *result);

    *result = cusolverDnDgetrs(resource_mg_get_or_null(&rm_cusolver, (void*)handle),
                               (cublasOperation_t)trans, n, nrhs,
                               mem_ptr_A,
                               lda,
                               mem_ptr_devIpiv,
                               mem_ptr_B,
                               ldb,
                               mem_ptr_devInfo);

    LOGE(LOG_DEBUG, "handle: %p, A: %p, devIpiv: %p, B: %p, devInfo: %p", handle, A, devIpiv, B, devInfo);
    LOGE(LOG_DEBUG, "trans: %d, n: %d, nrhs: %d, lda: %d, ldb: %d, result: %d", trans, n, nrhs, lda, ldb, *result);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cusolverdndestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "cusolverDnDestroy");
    *result = cusolverDnDestroy(resource_mg_get_or_null(&rm_cusolver, (void*)handle));
    RECORD_RESULT(integer, *result);
    GSCHED_RELEASE;
    return 1;
}
