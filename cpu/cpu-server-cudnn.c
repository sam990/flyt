
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <stdbool.h>
#include <string.h>
#include <cuda_runtime.h>

#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#include "gsched.h"
#include "cpu-server-client-mgr.h"

#define WITH_RECORDER
#include "api-recorder.h"

#include "cpu-server-cudnn.h"



int server_cudnn_init(int bypass)
{
    int ret = 0;
    ret &= resource_mg_init(&rm_cudnn, bypass);
    ret &= resource_mg_init(&rm_cudnn_tensors, bypass);
    ret &= resource_mg_init(&rm_cudnn_filters, bypass);
    ret &= resource_mg_init(&rm_cudnn_poolings, bypass);
    ret &= resource_mg_init(&rm_cudnn_activations, bypass);
    ret &= resource_mg_init(&rm_cudnn_lrns, bypass);
    ret &= resource_mg_init(&rm_cudnn_convs, bypass);
    ret &= resource_mg_init(&rm_cudnn_backendds, bypass);
    return ret;
}

int server_cudnn_deinit(void)
{
    resource_mg_free(&rm_cudnn);
    resource_mg_free(&rm_cudnn_tensors);
    resource_mg_free(&rm_cudnn_filters);
    resource_mg_free(&rm_cudnn_poolings);
    resource_mg_free(&rm_cudnn_activations);
    resource_mg_free(&rm_cudnn_lrns);
    resource_mg_free(&rm_cudnn_convs);
    resource_mg_free(&rm_cudnn_backendds);
    return 0;

}

bool_t rpc_cudnngetversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetmaxdeviceversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetMaxDeviceVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetcudartversion_1_svc(size_t *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnGetCudartVersion();
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngeterrorstring_1_svc(int status, char **result, struct svc_req *rqstp)
{
    const char* str;
    *result = malloc(128);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    str = cudnnGetErrorString((cudnnStatus_t)status);
    strncpy(*result, str, 128);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnqueryruntimeerror_1_svc(ptr handle, int mode, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    cudnnRuntimeTag_t *tag;
    
    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnQueryRuntimeError(
        GET_CUDNN_HANDLE(handle),
        (cudnnStatus_t*)&result->int_result_u.data, (cudnnErrQueryMode_t)mode, tag);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetproperty_1_svc(int type, int_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    result->err = cudnnGetProperty((libraryPropertyType)type, &result->int_result_u.data); 
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreate_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    cudnnHandle_t handle;

    GSCHED_RETAIN;
    result->err = cudnnCreate(&handle);
    if (result->err == CUDNN_STATUS_SUCCESS) {
        if (resource_map_add(client->rm_cudnn, (void*)handle, NULL, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }
    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnndestroy_1_svc(ptr handle, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(handle);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnDestroy(GET_CUDNN_HANDLE(handle));

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_handle(client, handle);
    }
    
    // TODO: Remove from resource manager
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetstream_1_svc(ptr handle, ptr streamId, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetstream_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(streamId);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;

    GET_CLIENT_CUDNN(*result);

    void *stream_ptr;

    GET_STREAM(stream_ptr, streamId, *result);

    *result = cudnnSetStream(
        GET_CUDNN_HANDLE(handle),
        (cudaStream_t)stream_ptr);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetstream_1_svc(ptr handle, ptr_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnGetStream(
        GET_CUDNN_HANDLE(handle),
        (cudaStream_t*)&result->ptr_result_u.ptr);

    // Need to do a reverse search to get the stream id
    if (result->err == CUDNN_STATUS_SUCCESS) {
        result->ptr_result_u.ptr = resource_map_get_key(client->custom_streams, result->ptr_result_u.ptr);
    }

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreatetensordescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    cudnnTensorDescriptor_t desc;

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnCreateTensorDescriptor(&desc);
    
    if (result->err == CUDNN_STATUS_SUCCESS) {
        cudnn_tensor_desc_args_t *args = (cudnn_tensor_desc_args_t *)malloc(sizeof(cudnn_tensor_desc_args_t));
        memset(args, 0, sizeof(cudnn_tensor_desc_args_t));
        if (resource_map_add(client->rm_cudnn_tensors, (void*)desc, args, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }

    GSCHED_RELEASE;
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsettensor4ddescriptor_1_svc(ptr tensorDesc, int format, int dataType, int n, int c, int h, int w, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensor4ddescriptor_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(format);
    RECORD_NARG(dataType);
    RECORD_NARG(n);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);


    GSCHED_RETAIN;
    *result = cudnnSetTensor4dDescriptor(
        GET_CUDNN_TENSOR(tensorDesc),
        (cudnnTensorFormat_t)format,
        (cudnnDataType_t)dataType,
        n, c, h, w);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_tensor_desc_args_t *args = (cudnn_tensor_desc_args_t *)resource_map_get(client->rm_cudnn_tensors, (void*)tensorDesc)->args;
        args->type = CUDNN_TENSOR_DESC_TYPE_4D;
        args->dataType = dataType;
        args->nbDims = 4;
        args->dims[0] = n;
        args->dims[1] = c;
        args->dims[2] = h;
        args->dims[3] = w;
    }


    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsettensor4ddescriptorex_1_svc(ptr tensorDesc, int dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensor4ddescriptorex_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(n);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);
    RECORD_NARG(nStride);
    RECORD_NARG(cStride);
    RECORD_NARG(hStride);
    RECORD_NARG(wStride);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetTensor4dDescriptorEx(
        GET_CUDNN_TENSOR(tensorDesc),
        (cudnnDataType_t)dataType,
        n, c, h, w, nStride, cStride, hStride, wStride);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_tensor_desc_args_t *args = (cudnn_tensor_desc_args_t *)resource_map_get(client->rm_cudnn_tensors, (void*)tensorDesc)->args;
        args->type = CUDNN_TENSOR_DESC_TYPE_4D_EX;
        args->dataType = dataType;
        args->nbDims = 4;
        args->dims[0] = n;
        args->dims[1] = c;
        args->dims[2] = h;
        args->dims[3] = w;
        args->strides[0] = nStride;
        args->strides[1] = cStride;
        args->strides[2] = hStride;
        args->strides[3] = wStride;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngettensor4ddescriptor_1_svc(ptr tensorDesc, int9_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnGetTensor4dDescriptor(
        GET_CUDNN_TENSOR(tensorDesc),
        (cudnnDataType_t*)&result->int9_result_u.data[0],
        &result->int9_result_u.data[1],
        &result->int9_result_u.data[2],
        &result->int9_result_u.data[3],
        &result->int9_result_u.data[4],
        &result->int9_result_u.data[5],
        &result->int9_result_u.data[6],
        &result->int9_result_u.data[7],
        &result->int9_result_u.data[8]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsettensornddescriptor_1_svc(ptr tensorDesc, int dataType, int nbDims, mem_data dimA, mem_data strideA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensornddescriptor_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(nbDims);
    RECORD_NARG(dimA);
    RECORD_NARG(strideA);
    
    //TODO: Recording dimA and strideA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    if (dimA.mem_data_len != nbDims * sizeof(int) || strideA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetTensorNdDescriptor(
        GET_CUDNN_TENSOR(tensorDesc),
        (cudnnDataType_t)dataType,
        nbDims,
        (const int*)dimA.mem_data_val,
        (const int*)strideA.mem_data_val);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_tensor_desc_args_t *args = (cudnn_tensor_desc_args_t *)resource_map_get(client->rm_cudnn_tensors, (void*)tensorDesc)->args;
        args->type = CUDNN_TENSOR_DESC_TYPE_ND;
        args->dataType = dataType;
        args->nbDims = nbDims;
        memcpy(args->dims, dimA.mem_data_val, nbDims*sizeof(int));
        memcpy(args->strides, strideA.mem_data_val, nbDims*sizeof(int));
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsettensornddescriptorex_1_svc(ptr tensorDesc, int format, int dataType, int nbDims, mem_data dimA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsettensornddescriptorex_1_argument);
    RECORD_NARG(tensorDesc);
    RECORD_NARG(format);
    RECORD_NARG(dataType);
    RECORD_NARG(nbDims);
    RECORD_NARG(dimA);
    
    //TODO: Recording dimA and strideA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    if (dimA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetTensorNdDescriptorEx(
        GET_CUDNN_TENSOR(tensorDesc),
        (cudnnTensorFormat_t)format,   
        (cudnnDataType_t)dataType,
        nbDims,
        (const int*)dimA.mem_data_val);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_tensor_desc_args_t *args = (cudnn_tensor_desc_args_t *)resource_map_get(client->rm_cudnn_tensors, (void*)tensorDesc)->args;
        args->type = CUDNN_TENSOR_DESC_TYPE_ND_EX;
        args->format = format;
        args->dataType = dataType;
        args->nbDims = nbDims;
        memcpy(args->dims, dimA.mem_data_val, nbDims*sizeof(int));
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngettensornddescriptor_1_svc(ptr tensorDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnDataType_t) + sizeof(int) + nbDimsRequested*sizeof(int)*2;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }

    
    GSCHED_RETAIN;
    result->err = cudnnGetTensorNdDescriptor(
        GET_CUDNN_TENSOR(tensorDesc),
        nbDimsRequested,
        (cudnnDataType_t*)result->mem_result_u.data.mem_data_val,
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(int)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(int)+nbDimsRequested*sizeof(int)]);

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngettensorsizeinbytes_1_svc(ptr tensorDesc, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnGetTensorSizeInBytes(
        GET_CUDNN_TENSOR(tensorDesc),
        &result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroytensordescriptor_1_svc(ptr tensorDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(tensorDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnDestroyTensorDescriptor(
        GET_CUDNN_TENSOR(tensorDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_tensor(client, tensorDesc);
    }

    RECORD_RESULT(integer, *result);
    return 1;
}


bool_t rpc_cudnncreatefilterdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    cudnnFilterDescriptor_t desc;

    GSCHED_RETAIN;
    result->err = cudnnCreateFilterDescriptor(&desc);
    GSCHED_RELEASE;

    if (result->err == CUDNN_STATUS_SUCCESS) {
        cudnn_filter_desc_args_t *args = (cudnn_filter_desc_args_t *)malloc(sizeof(cudnn_filter_desc_args_t));
        memset(args, 0, sizeof(cudnn_filter_desc_args_t));
        if (resource_map_add(client->rm_cudnn_filters, (void*)desc, args, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetfilter4ddescriptor_1_svc(ptr filterDesc, int dataType, int format, int k, int c, int h, int w, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetfilter4ddescriptor_1_argument);
    RECORD_NARG(filterDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(format);
    RECORD_NARG(k);
    RECORD_NARG(c);
    RECORD_NARG(h);
    RECORD_NARG(w);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetFilter4dDescriptor(
        GET_CUDNN_FILTER(filterDesc),
        (cudnnDataType_t)dataType,
        (cudnnTensorFormat_t)format,
        k, c, h, w);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_filter_desc_args_t *args = (cudnn_filter_desc_args_t *)resource_map_get(client->rm_cudnn_filters, (void*)filterDesc)->args;
        args->type = CUDNN_FILTER_DESC_TYPE_4D;
        args->dataType = dataType;
        args->format = format;
        args->nbDims = 4;
        args->dims[0] = k;
        args->dims[1] = c;
        args->dims[2] = h;
        args->dims[3] = w;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetfilter4ddescriptor_1_svc(ptr filterDesc, int6_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnGetFilter4dDescriptor(
        GET_CUDNN_FILTER(filterDesc),
        (cudnnDataType_t*)&result->int6_result_u.data[0],
        (cudnnTensorFormat_t*)&result->int6_result_u.data[1],
        &result->int6_result_u.data[2],
        &result->int6_result_u.data[3],
        &result->int6_result_u.data[4],
        &result->int6_result_u.data[5]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetfilternddescriptor_1_svc(ptr filterDesc, int dataType, int format, int nbDims, mem_data filterDimA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetfilternddescriptor_1_argument);
    RECORD_NARG(filterDesc);
    RECORD_NARG(dataType);
    RECORD_NARG(format);
    RECORD_NARG(nbDims);
    RECORD_NARG(filterDimA);
    
    //TODO: Recording filterDimA is not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    if (filterDimA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimension not as expected.");
        return 0;
    }

    GET_CLIENT_CUDNN(*result);


    GSCHED_RETAIN;
    *result = cudnnSetFilterNdDescriptor(
        GET_CUDNN_FILTER(filterDesc),
        (cudnnDataType_t)dataType,
        (cudnnTensorFormat_t)format,
        nbDims,
        (const int*)filterDimA.mem_data_val);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_filter_desc_args_t *args = (cudnn_filter_desc_args_t *)resource_map_get(client->rm_cudnn_filters, (void*)filterDesc)->args;
        args->type = CUDNN_FILTER_DESC_TYPE_ND;
        args->dataType = dataType;
        args->format = format;
        args->nbDims = nbDims;
        memcpy(args->dims, filterDimA.mem_data_val, nbDims*sizeof(int));
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetfilternddescriptor_1_svc(ptr filterDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    
    result->mem_result_u.data.mem_data_len = sizeof(cudnnDataType_t) + sizeof(cudnnTensorFormat_t) + sizeof(int) + nbDimsRequested*sizeof(int);
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }


    
    GSCHED_RETAIN;
    result->err = cudnnGetFilterNdDescriptor(
        GET_CUDNN_FILTER(filterDesc),
        nbDimsRequested,
        (cudnnDataType_t*)result->mem_result_u.data.mem_data_val,
        (cudnnTensorFormat_t*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(cudnnTensorDescriptor_t)],
        (int*)&result->mem_result_u.data.mem_data_val[sizeof(cudnnDataType_t)+sizeof(cudnnTensorDescriptor_t)+sizeof(int)]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetfiltersizeinbytes_1_svc(ptr filterDesc, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    GSCHED_RETAIN;
    result->err = cudnnGetFilterSizeInBytes(
        GET_CUDNN_FILTER(filterDesc),
        &result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnntransformfilter_1_svc(ptr handle, ptr transDesc, cudnn_scaling_t alpha, ptr srcDesc, ptr srcData, cudnn_scaling_t beta, ptr destDesc, ptr destData, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnntransformfilter_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(transDesc);
    RECORD_NARG(alpha);
    RECORD_NARG(srcDesc);
    RECORD_NARG(srcData);
    RECORD_NARG(beta);
    RECORD_NARG(destDesc);
    RECORD_NARG(destData);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnTransformFilter(
        GET_CUDNN_HANDLE(handle),
        GET_CUDNN_TENSOR_TRANSFORM(transDesc),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_FILTER(srcDesc),
        (const void*)srcData,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_FILTER(destDesc),
        (void*)destData);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnndestroyfilterdescriptor_1_svc(ptr filterDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(filterDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnDestroyFilterDescriptor(
        GET_CUDNN_FILTER(filterDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_filter(client, filterDesc);
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreatepoolingdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    cudnnPoolingDescriptor_t desc;

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnCreatePoolingDescriptor(&desc);
    GSCHED_RELEASE;

    if (result->err == CUDNN_STATUS_SUCCESS) {
        cudnn_pooling_desc_args_t *args = (cudnn_pooling_desc_args_t *)malloc(sizeof(cudnn_pooling_desc_args_t));
        memset(args, 0, sizeof(cudnn_pooling_desc_args_t));
        if (resource_map_add(client->rm_cudnn_poolings, (void*)desc, args, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetpooling2ddescriptor_1_svc(ptr poolingDesc, int mode, int maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetpooling2ddescriptor_1_argument);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(mode);
    RECORD_NARG(maxpoolingNanOpt);
    RECORD_NARG(windowHeight);
    RECORD_NARG(windowWidth);
    RECORD_NARG(verticalPadding);
    RECORD_NARG(horizontalPadding);
    RECORD_NARG(verticalStride);
    RECORD_NARG(horizontalStride);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetPooling2dDescriptor(
        GET_CUDNN_POOLING(poolingDesc),
        (cudnnPoolingMode_t)mode,
        (cudnnNanPropagation_t)maxpoolingNanOpt,
        windowHeight, windowWidth,
        verticalPadding, horizontalPadding,
        verticalStride, horizontalStride);
    GSCHED_RELEASE;

    if (*result = CUDNN_STATUS_SUCCESS) {
        cudnn_pooling_desc_args_t *args = (cudnn_pooling_desc_args_t *)resource_map_get(client->rm_cudnn_poolings, (void*)poolingDesc)->args;
        args->type = CUDNN_POOLING_DESC_TYPE_2D;
        args->mode = mode;
        args->maxpoolingNanOpt = maxpoolingNanOpt;
        args->windowDimA[0] = windowHeight;
        args->windowDimA[1] = windowWidth;
        args->paddingA[0] = verticalPadding;
        args->paddingA[1] = horizontalPadding;
        args->strideA[0] = verticalStride;
        args->strideA[1] = horizontalStride;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetpooling2ddescriptor_1_svc(ptr poolingDesc, int8_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnGetPooling2dDescriptor(
        GET_CUDNN_POOLING(poolingDesc),
        (cudnnPoolingMode_t*)&result->int8_result_u.data[0],
        (cudnnNanPropagation_t*)&result->int8_result_u.data[1],
        &result->int8_result_u.data[2],
        &result->int8_result_u.data[3],
        &result->int8_result_u.data[4],
        &result->int8_result_u.data[5],
        &result->int8_result_u.data[6],
        &result->int8_result_u.data[7]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetpoolingnddescriptor_1_svc(ptr poolingDesc, int mode, int maxpoolingNanOpt, int nbDims, mem_data windowDimA, mem_data paddingA, mem_data strideA, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetpoolingnddescriptor_1_argument);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(mode);
    RECORD_NARG(maxpoolingNanOpt);
    RECORD_NARG(nbDims);
    RECORD_NARG(windowDimA);
    RECORD_NARG(paddingA);
    RECORD_NARG(strideA);
    //TODO: Recording windowDimA, paddingA and strideA are not as easy as done here.

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    if (windowDimA.mem_data_len != nbDims * sizeof(int) ||
        paddingA.mem_data_len != nbDims * sizeof(int) ||
        strideA.mem_data_len != nbDims * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    GSCHED_RETAIN;
    *result = cudnnSetPoolingNdDescriptor(
        GET_CUDNN_POOLING(poolingDesc),
        (cudnnPoolingMode_t)mode,
        (cudnnNanPropagation_t)maxpoolingNanOpt,
        nbDims,
        (const int*)windowDimA.mem_data_val,
        (const int*)paddingA.mem_data_val,
        (const int*)strideA.mem_data_val);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_pooling_desc_args_t *args = (cudnn_pooling_desc_args_t *)resource_map_get(client->rm_cudnn_poolings, (void*)poolingDesc)->args;
        args->type = CUDNN_POOLING_DESC_TYPE_ND;
        args->mode = mode;
        args->maxpoolingNanOpt = maxpoolingNanOpt;
        args->nbDims = nbDims;
        memcpy(args->windowDimA, windowDimA.mem_data_val, nbDims*sizeof(int));
        memcpy(args->paddingA, paddingA.mem_data_val, nbDims*sizeof(int));
        memcpy(args->strideA, strideA.mem_data_val, nbDims*sizeof(int));
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetpoolingnddescriptor_1_svc(ptr poolingDesc, int nbDimsRequested, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    result->mem_result_u.data.mem_data_len = sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + nbDimsRequested * sizeof(int) * 3;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    
    size_t offsets[] = {
        0,
        sizeof(cudnnPoolingMode_t),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int),
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int) + sizeof(int) * nbDimsRequested,
        sizeof(cudnnPoolingMode_t) + sizeof(cudnnNanPropagation_t) + sizeof(int) + sizeof(int) * nbDimsRequested * 2,
    };
    
    GSCHED_RETAIN;
    result->err = cudnnGetPoolingNdDescriptor(
        GET_CUDNN_POOLING(poolingDesc),
        nbDimsRequested,
        (cudnnPoolingMode_t*)result->mem_result_u.data.mem_data_val[offsets[0]],
        (cudnnNanPropagation_t*)result->mem_result_u.data.mem_data_val[offsets[1]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[2]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[3]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[4]],
        (int*)result->mem_result_u.data.mem_data_val[offsets[5]]);

    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetpoolingndforwardoutputdim_1_svc(ptr poolingDesc, ptr inputTensorDesc, int nbDims, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    GSCHED_RETAIN;
    result->mem_result_u.data.mem_data_len = sizeof(int) * nbDims;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    result->err = cudnnGetPoolingNdForwardOutputDim(
        GET_CUDNN_POOLING(poolingDesc),
        GET_CUDNN_TENSOR(inputTensorDesc),
        nbDims,
        (int*)result->mem_result_u.data.mem_data_val);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetpooling2dforwardoutputdim_1_svc(ptr poolingDesc, ptr inputTensorDesc, int4_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    GSCHED_RETAIN;
    result->err = cudnnGetPooling2dForwardOutputDim(
        GET_CUDNN_POOLING(poolingDesc),
        GET_CUDNN_TENSOR(inputTensorDesc),
        (int*)&result->int4_result_u.data[0],
        (int*)&result->int4_result_u.data[1],
        (int*)&result->int4_result_u.data[2],
        (int*)&result->int4_result_u.data[3]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroypoolingdescriptor_1_svc(ptr poolingDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(poolingDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnDestroyPoolingDescriptor(
        GET_CUDNN_POOLING(poolingDesc));
    // TODO: Remove from resource manager
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_pooling(client, poolingDesc);
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreateactivationdescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    cudnnActivationDescriptor_t desc;

    GSCHED_RETAIN;
    result->err = cudnnCreateActivationDescriptor(&desc);
    GSCHED_RELEASE;
    if (result->err == CUDNN_STATUS_SUCCESS) {
        cudnn_activation_desc_args_t *args = (cudnn_activation_desc_args_t *)malloc(sizeof(cudnn_activation_desc_args_t));
        memset(args, 0, sizeof(cudnn_activation_desc_args_t));
        if (resource_map_add(&rm_cudnn_activations, (void*)desc, args, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }
    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetactivationdescriptor_1_svc(ptr activationDesc, int mode, int reluNanOpt, double coef, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetactivationdescriptor_1_argument);
    RECORD_NARG(activationDesc);
    RECORD_NARG(mode);
    RECORD_NARG(reluNanOpt);
    RECORD_NARG(coef);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetActivationDescriptor(
        (cudnnActivationDescriptor_t)resource_mg_get_or_null(&rm_cudnn_activations, (void*)activationDesc),
        (cudnnActivationMode_t)mode,
        (cudnnNanPropagation_t)reluNanOpt,
        coef);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_activation_desc_args_t *args = (cudnn_activation_desc_args_t *)resource_map_get(client->rm_cudnn_activations, (void*)activationDesc)->args;
        args->activateSet = 1;
        args->mode = mode;
        args->reluNanOpt = reluNanOpt;
        args->coef = coef;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetactivationdescriptor_1_svc(ptr activationDesc, int2d1_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    GSCHED_RETAIN;
    result->err = cudnnGetActivationDescriptor(
        GET_CUDNN_ACTIVATION(activationDesc),
        (cudnnActivationMode_t*)&result->int2d1_result_u.data.i[0],
        (cudnnNanPropagation_t*)&result->int2d1_result_u.data.i[1],
        &result->int2d1_result_u.data.d);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnsetactivationdescriptorswishbeta_1_svc(ptr activationDesc, double swish_beta, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetactivationdescriptorswishbeta_1_argument);
    RECORD_NARG(activationDesc);
    RECORD_NARG(swish_beta);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetActivationDescriptorSwishBeta(
        GET_CUDNN_ACTIVATION(activationDesc),
        swish_beta);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_activation_desc_args_t *args = (cudnn_activation_desc_args_t *)resource_map_get(client->rm_cudnn_activations, (void*)activationDesc)->args;
        args->swishBetaSet = 1;
        args->swishBeta = swish_beta;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetactivationdescriptorswishbeta_1_svc(ptr activationDesc, d_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnGetActivationDescriptorSwishBeta(
        GET_CUDNN_ACTIVATION(activationDesc),
        &result->d_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroyactivationdescriptor_1_svc(ptr activationDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(activationDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(*result);
    GSCHED_RETAIN;
    *result = cudnnDestroyActivationDescriptor(
        GET_CUDNN_ACTIVATION(activationDesc));
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_activation(client, activationDesc);
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnncreatelrndescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    cudnnLRNDescriptor_t desc;

    GSCHED_RETAIN;
    result->err = cudnnCreateLRNDescriptor(&desc);
    GSCHED_RELEASE;

    if (result->err == CUDNN_STATUS_SUCCESS) {
        cudnn_lrn_desc_args_t *args = (cudnn_lrn_desc_args_t *)malloc(sizeof(cudnn_lrn_desc_args_t));
        memset(args, 0, sizeof(cudnn_lrn_desc_args_t));
        if (resource_map_add(&rm_cudnn_lrns, (void*)desc, args, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnsetlrndescriptor_1_svc(ptr normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetlrndescriptor_1_argument);
    RECORD_NARG(normDesc);
    RECORD_NARG(lrnN);
    RECORD_NARG(lrnAlpha);
    RECORD_NARG(lrnBeta);
    RECORD_NARG(lrnK);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetLRNDescriptor(
        GET_CUDNN_LRN(normDesc),
        lrnN,
        lrnAlpha,
        lrnBeta,
        lrnK);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_lrn_desc_args_t *args = (cudnn_lrn_desc_args_t *)resource_map_get(client->rm_cudnn_lrns, (void*)normDesc)->args;
        args->lrnN = lrnN;
        args->lrnAlpha = lrnAlpha;
        args->lrnBeta = lrnBeta;
        args->lrnK = lrnK;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetlrndescriptor_1_svc(ptr normDesc, int1d3_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    GSCHED_RETAIN;
    result->err = cudnnGetLRNDescriptor(
        GET_CUDNN_LRN(normDesc),
        (unsigned int*)&result->int1d3_result_u.data.i,
        &result->int1d3_result_u.data.d[0],
        &result->int1d3_result_u.data.d[1],
        &result->int1d3_result_u.data.d[2]);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnndestroylrndescriptor_1_svc(ptr lrnDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(lrnDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);
    GSCHED_RETAIN;
    *result = cudnnDestroyLRNDescriptor(
        GET_CUDNN_LRN(lrnDesc));
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_lrn(client, lrnDesc);
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnpoolingforward_1_svc(ptr handle, ptr poolingDesc, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnpoolingforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(poolingDesc);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;

    GET_CLIENT_CUDNN(*result)

    *result = cudnnPoolingForward(
        GET_CUDNN_HANDLE(handle),
        GET_CUDNN_POOLING(poolingDesc),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(xDesc),
        (const void*)x,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(yDesc),
        (void*)y);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnactivationforward_1_svc(ptr handle, ptr activationDesc, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnactivationforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(activationDesc);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;

    GET_CLIENT_CUDNN(*result)

    *result = cudnnActivationForward(
        GET_CUDNN_HANDLE(handle),
        GET_CUDNN_ACTIVATION(activationDesc),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(xDesc),
        (const void*)x,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(yDesc),
        (void*)y);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnlrncrosschannelforward_1_svc(ptr handle, ptr normDesc, int lrnMode, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnlrncrosschannelforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(normDesc);
    RECORD_NARG(lrnMode);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;

    GET_CLIENT_CUDNN(*result)

    *result = cudnnLRNCrossChannelForward(
        GET_CUDNN_HANDLE(handle),
        GET_CUDNN_LRN(normDesc),
        (cudnnLRNMode_t)lrnMode,
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(xDesc),
        (const void*)x,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(yDesc),
        (void*)y);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsoftmaxforward_1_svc(ptr handle, int algo, int mode, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsoftmaxforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(algo);
    RECORD_NARG(mode);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;

    GET_CLIENT_CUDNN(*result)

    *result = cudnnSoftmaxForward(
        GET_CUDNN_HANDLE(handle),
        (cudnnSoftmaxAlgorithm_t)algo,
        (cudnnSoftmaxMode_t)mode,
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(xDesc),
        (const void*)x,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(yDesc),
        (void*)y);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

/* cudnn cnn inference */
bool_t rpc_cudnngetconvolutionndforwardoutputdim_1_svc(ptr convDesc, ptr inputTensorDesc, ptr filterDesc, int nbDims, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    result->mem_result_u.data.mem_data_len = sizeof(int) * nbDims;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    GSCHED_RETAIN;
    result->err = cudnnGetConvolutionNdForwardOutputDim(
        GET_CUDNN_CONV(convDesc),
        GET_CUDNN_TENSOR(inputTensorDesc),
        GET_CUDNN_FILTER(filterDesc),
        nbDims,
        (int*)result->mem_result_u.data.mem_data_val);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnncreateconvolutiondescriptor_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    RECORD_VOID_API;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    cudnnConvolutionDescriptor_t desc;
    
    GSCHED_RETAIN;
    result->err = cudnnCreateConvolutionDescriptor(&desc);
    GSCHED_RELEASE;

    if (result->err == CUDNN_STATUS_SUCCESS) {
        cudnn_conv_desc_args_t *args = (cudnn_conv_desc_args_t *)malloc(sizeof(cudnn_conv_desc_args_t));
        memset(args, 0, sizeof(cudnn_conv_desc_args_t));
        if (resource_map_add(&rm_cudnn_convs, (void*)desc, args, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnndestroyconvolutiondescriptor_1_svc(ptr convDesc, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(convDesc);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnDestroyConvolutionDescriptor(
        GET_CUDNN_CONV(convDesc));
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_conv(client, convDesc);
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetconvolutionmathtype_1_svc(ptr convDesc, int mathType, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetconvolutionmathtype_1_argument);
    RECORD_NARG(convDesc);
    RECORD_NARG(mathType);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetConvolutionMathType(
        GET_CUDNN_CONV(convDesc),
        (cudnnMathType_t)mathType);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_conv_desc_args_t *args = (cudnn_conv_desc_args_t *)resource_map_get(client->rm_cudnn_convs, (void*)convDesc)->args;
        args->mathType = mathType;
        args->mathTypeSet = 1;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}


bool_t rpc_cudnnsetconvolutiongroupcount_1_svc(ptr convDesc, int groupCount, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetconvolutiongroupcount_1_argument);
    RECORD_NARG(convDesc);
    RECORD_NARG(groupCount);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnSetConvolutionGroupCount(
        GET_CUDNN_CONV(convDesc),
        groupCount);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_conv_desc_args_t *args = (cudnn_conv_desc_args_t *)resource_map_get(client->rm_cudnn_convs, (void*)convDesc)->args;
        args->groupCount = groupCount;
        args->groupCountSet = 1;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnsetconvolutionnddescriptor_1_svc(ptr convDesc, int arrayLength, mem_data padA, mem_data filterStrideA, mem_data dilationA, int mode, int computeType, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnsetconvolutionnddescriptor_1_argument);
    RECORD_NARG(convDesc);
    RECORD_NARG(arrayLength);
    RECORD_NARG(padA);
    RECORD_NARG(filterStrideA);
    RECORD_NARG(dilationA);
    RECORD_NARG(mode);
    RECORD_NARG(computeType);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    if (padA.mem_data_len != arrayLength * sizeof(int) ||
        filterStrideA.mem_data_len != arrayLength * sizeof(int) ||
        dilationA.mem_data_len != arrayLength * sizeof(int)) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }

    GSCHED_RETAIN;
    *result = cudnnSetConvolutionNdDescriptor(
        GET_CUDNN_CONV(convDesc),
        arrayLength,
        (const int*)padA.mem_data_val,
        (const int*)filterStrideA.mem_data_val,
        (const int*)dilationA.mem_data_val,
        (cudnnConvolutionMode_t)mode,
        (cudnnDataType_t)computeType);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_conv_desc_args_t *args = (cudnn_conv_desc_args_t *)resource_map_get(client->rm_cudnn_convs, (void*)convDesc)->args;
        args->type = CUDNN_CONV_DESC_TYPE_ND;
        args->nbDims = arrayLength;
        memcpy(args->padA, padA.mem_data_val, arrayLength*sizeof(int));
        memcpy(args->filterStrideA, filterStrideA.mem_data_val, arrayLength*sizeof(int));
        memcpy(args->dilationA, dilationA.mem_data_val, arrayLength*sizeof(int));
        args->mode = mode;
        args->dataType = computeType;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnngetconvolutionforwardalgorithm_v7_1_svc(ptr handle, ptr srcDesc, ptr filterDesc, ptr convDesc, ptr destDesc, int requestedAlgoCount, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    result->mem_result_u.data.mem_data_len = sizeof(int) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    GSCHED_RETAIN;
    result->err = cudnnGetConvolutionForwardAlgorithm_v7(
        GET_CUDNN_HANDLE(handle),
        GET_CUDNN_TENSOR(srcDesc),
        GET_CUDNN_FILTER(filterDesc),
        GET_CUDNN_CONV(convDesc),
        GET_CUDNN_TENSOR(destDesc),
        requestedAlgoCount,
        (int*)result->mem_result_u.data.mem_data_val,
        (cudnnConvolutionFwdAlgoPerf_t*)(result->mem_result_u.data.mem_data_val + sizeof(int)));
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnfindconvolutionforwardalgorithm_1_svc(ptr handle, ptr xDesc, ptr wDesc, ptr convDesc, ptr yDesc, int requestedAlgoCount, mem_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    result->mem_result_u.data.mem_data_len = sizeof(int) + sizeof(cudnnConvolutionFwdAlgoPerf_t) * requestedAlgoCount;
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    GSCHED_RETAIN;
    result->err = cudnnFindConvolutionForwardAlgorithm(
        GET_CUDNN_HANDLE(handle),
        GET_CUDNN_TENSOR(xDesc),
        GET_CUDNN_FILTER(wDesc),
        GET_CUDNN_CONV(convDesc),
        GET_CUDNN_TENSOR(yDesc),
        requestedAlgoCount,
        (int*)result->mem_result_u.data.mem_data_val,
        (cudnnConvolutionFwdAlgoPerf_t*)(result->mem_result_u.data.mem_data_val + sizeof(int)));
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnngetconvolutionforwardworkspacesize_1_svc(ptr handle, ptr xDesc, ptr wDesc, ptr convDesc, ptr yDesc, int algo, sz_result *result, struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    GET_CLIENT_CUDNN(result->err);
    GSCHED_RETAIN;
    result->err = cudnnGetConvolutionForwardWorkspaceSize(
        GET_CUDNN_HANDLE(handle),
        GET_CUDNN_TENSOR(xDesc),
        GET_CUDNN_FILTER(wDesc),
        GET_CUDNN_CONV(convDesc),
        GET_CUDNN_TENSOR(yDesc),
        (cudnnConvolutionFwdAlgo_t)algo,
        (size_t*)&result->sz_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudnnconvolutionforward_1_svc(ptr handle, cudnn_scaling_t alpha, ptr xDesc, ptr x, ptr wDesc, ptr w, ptr convDesc, int algo, ptr workSpace, size_t workSpaceSizeInBytes, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnconvolutionforward_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(wDesc);
    RECORD_NARG(w);
    RECORD_NARG(convDesc);
    RECORD_NARG(algo);
    RECORD_NARG(workSpace);
    RECORD_NARG(workSpaceSizeInBytes);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;

    *result = cudnnConvolutionForward(
        GET_CUDNN_HANDLE(handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(xDesc),
        (const void*)x,
        GET_CUDNN_FILTER(wDesc),
        (const void*)w,
        GET_CUDNN_CONV(convDesc),
        algo,
        (void*)workSpace,
        workSpaceSizeInBytes,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(yDesc),
        (void*)y);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnaddtensor_1_svc(ptr handle, cudnn_scaling_t alpha, ptr aDesc, ptr A, cudnn_scaling_t beta, ptr cDesc, ptr C, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnaddtensor_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(alpha);
    RECORD_NARG(aDesc);
    RECORD_NARG(A);
    RECORD_NARG(beta);
    RECORD_NARG(cDesc);
    RECORD_NARG(C);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result)

    GSCHED_RETAIN;
    *result = cudnnAddTensor(
        GET_CUDNN_HANDLE(handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(aDesc),
        (const void*)A,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(cDesc),
        (void*)C);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnntransformtensor_1_svc(ptr handle, cudnn_scaling_t alpha, ptr xDesc, ptr x, cudnn_scaling_t beta, ptr yDesc, ptr y, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnntransformtensor_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(alpha);
    RECORD_NARG(xDesc);
    RECORD_NARG(x);
    RECORD_NARG(beta);
    RECORD_NARG(yDesc);
    RECORD_NARG(y);
    
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result)

    GSCHED_RETAIN;
    *result = cudnnTransformTensor(
        GET_CUDNN_HANDLE(handle),
        (alpha.dataType == CUDNN_DATA_DOUBLE ? (const void*)&alpha.cudnn_scaling_t_u.d : (const void*)&alpha.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(xDesc),
        (const void*)x,
        (beta.dataType == CUDNN_DATA_DOUBLE ? (const void*)&beta.cudnn_scaling_t_u.d : (const void*)&beta.cudnn_scaling_t_u.f),
        GET_CUDNN_TENSOR(yDesc),
        (void*)y);
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

static const size_t backendAttributeSizes[] = {
    [CUDNN_TYPE_HANDLE] = sizeof(cudnnHandle_t),
    [CUDNN_TYPE_DATA_TYPE] = sizeof(cudnnDataType_t),
    [CUDNN_TYPE_BOOLEAN] = sizeof(bool),
    [CUDNN_TYPE_INT64] = sizeof(int64_t),
    [CUDNN_TYPE_FLOAT] = sizeof(float),
    [CUDNN_TYPE_DOUBLE] = sizeof(double),
    [CUDNN_TYPE_VOID_PTR] = sizeof(void *),
    [CUDNN_TYPE_CONVOLUTION_MODE] = sizeof(cudnnConvolutionMode_t),
    [CUDNN_TYPE_HEUR_MODE] = sizeof(cudnnBackendHeurMode_t),
    [CUDNN_TYPE_KNOB_TYPE] = sizeof(cudnnBackendKnobType_t),
    [CUDNN_TYPE_NAN_PROPOGATION] = sizeof(cudnnNanPropagation_t),
    [CUDNN_TYPE_NUMERICAL_NOTE] = sizeof(cudnnBackendNumericalNote_t),
    [CUDNN_TYPE_LAYOUT_TYPE] = sizeof(cudnnBackendLayoutType_t),
    [CUDNN_TYPE_ATTRIB_NAME] = sizeof(cudnnBackendAttributeName_t),
    [CUDNN_TYPE_POINTWISE_MODE] = sizeof(cudnnPointwiseMode_t),
    [CUDNN_TYPE_BACKEND_DESCRIPTOR] = sizeof(cudnnBackendDescriptor_t),
    [CUDNN_TYPE_GENSTATS_MODE] = sizeof(cudnnGenStatsMode_t),
    [CUDNN_TYPE_BN_FINALIZE_STATS_MODE] = sizeof(cudnnBnFinalizeStatsMode_t),
    [CUDNN_TYPE_REDUCTION_OPERATOR_TYPE] = sizeof(cudnnReduceTensorOp_t),
    [CUDNN_TYPE_BEHAVIOR_NOTE] = sizeof(cudnnBackendBehaviorNote_t),
    [CUDNN_TYPE_TENSOR_REORDERING_MODE] = sizeof(cudnnBackendTensorReordering_t),
    [CUDNN_TYPE_RESAMPLE_MODE] = sizeof(cudnnResampleMode_t),
    [CUDNN_TYPE_PADDING_MODE] = sizeof(cudnnPaddingMode_t),
    [CUDNN_TYPE_INT32] = sizeof(int32_t),
    [CUDNN_TYPE_CHAR] = sizeof(char),
    [CUDNN_TYPE_SIGNAL_MODE] = sizeof(cudnnSignalMode_t),
    [CUDNN_TYPE_FRACTION] = sizeof(cudnnFraction_t),
    [CUDNN_TYPE_NORM_MODE] = sizeof(cudnnBackendNormMode_t),
    [CUDNN_TYPE_NORM_FWD_PHASE] = sizeof(cudnnBackendNormFwdPhase_t),
    [CUDNN_TYPE_RNG_DISTRIBUTION] = sizeof(cudnnRngDistribution_t),
};

bool_t rpc_cudnnbackendcreatedescriptor_1_svc(int descriptorType, ptr_result *result, struct svc_req *rqstp)
{
    RECORD_API(int);
    RECORD_SINGLE_ARG(descriptorType);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(result->err);

    cudnnBackendDescriptor_t desc;

    GSCHED_RETAIN;
    result->err = cudnnBackendCreateDescriptor(
        (cudnnBackendDescriptorType_t)descriptorType,
        (cudnnBackendDescriptor_t*)&desc);
    GSCHED_RELEASE;

    if (result->err == CUDNN_STATUS_SUCCESS) {
        cudnn_backend_desc_args_t *args = (cudnn_backend_desc_args_t *)malloc(sizeof(cudnn_backend_desc_args_t));
        args->descriptorType = descriptorType;
        memset(args, 0, sizeof(cudnn_backend_desc_args_t));
        list_init(&args->attributes, sizeof(cudnn_backend_attr_t));
        if (resource_map_add(&rm_cudnn_backendds, (void*)desc, args, (void**)&result->ptr_result_u.ptr) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
        }
    }

    RECORD_RESULT(ptr_result_u, *result);
    return 1;
}

bool_t rpc_cudnnbackenddestroydescriptor_1_svc(ptr descriptor, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(descriptor);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnBackendDestroyDescriptor(GET_CUDNN_BACKEND(descriptor));
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        free_client_cudnn_backend(client, descriptor);
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnbackendinitialize_1_svc(ptr descriptor, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(descriptor);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnBackendInitialize(
        GET_CUDNN_BACKEND(descriptor));
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_backend_desc_args_t *args = (cudnn_backend_desc_args_t *)resource_map_get(client->rm_cudnn_backends, (void*)descriptor)->args;
        args->initialized = 1;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnbackendfinalize_1_svc(ptr descriptor, int *result, struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(descriptor);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);

    GSCHED_RETAIN;
    *result = cudnnBackendFinalize(
        GET_CUDNN_BACKEND(descriptor));
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_backend_desc_args_t *args = (cudnn_backend_desc_args_t *)resource_map_get(client->rm_cudnn_backends, (void*)descriptor)->args;
        args->finalized = 1;
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

int map_cudnn_backend_attribute(cricket_client* client, int attributeType, int64_t elemCount, void *data, void *dest) {
    switch(attributeType) {
        case CUDNN_TYPE_HANDLE:
            for (int i = 0; i < elemCount; i++)
                ((cudnnHandle_t*)dest)[i] = GET_CUDNN_HANDLE(((ptr*)data)[i]);
            break;
        case CUDNN_TYPE_BACKEND_DESCRIPTOR:
            for (int i = 0; i < elemCount; i++)
                ((cudnnBackendDescriptor_t*)dest)[i] = GET_CUDNN_BACKEND(((ptr*)data)[i]);
            break;
        default:
            memcpy(dest, data, elemCount * backendAttributeSizes[attributeType]);
            break;
    }
    return 0;
}

bool_t rpc_cudnnbackendsetattribute_1_svc(
                         ptr descriptor,
                         int attributeName,
                         int attributeType,
                         int64_t elementCount,
                         mem_data arrayOfElements,
                         int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnbackendsetattribute_1_argument);
    RECORD_NARG(descriptor);
    RECORD_NARG(attributeName);
    RECORD_NARG(attributeType);
    RECORD_NARG(elementCount);
    RECORD_NARG(arrayOfElements);

    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GET_CLIENT_CUDNN(*result);
    
    if (attributeType < 0 || attributeType >= CUDNN_TYPE_RNG_DISTRIBUTION) {
        LOGE(LOG_ERROR, "attributeType out of range.");
        return 0;
    }

    if (arrayOfElements.mem_data_len != elementCount * backendAttributeSizes[attributeType]) {
        LOGE(LOG_ERROR, "array dimensions not as expected.");
        return 0;
    }
    
    uint8_t *attribData[arrayOfElements.mem_data_len];
    map_cudnn_backend_attribute(client, attributeType, elementCount, arrayOfElements.mem_data_val, attribData);

    GSCHED_RETAIN;
    *result = cudnnBackendSetAttribute(
        (cudnnBackendDescriptor_t)resource_mg_get_or_null(&rm_cudnn_backendds, (void*)descriptor),
        (cudnnBackendAttributeName_t)attributeName,
        (cudnnBackendAttributeType_t)attributeType,
        elementCount,
        attribData);
    GSCHED_RELEASE;

    if (*result == CUDNN_STATUS_SUCCESS) {
        cudnn_backend_desc_args_t *args = (cudnn_backend_desc_args_t *)resource_map_get(client->rm_cudnn_backends, (void*)descriptor)->args;
        
    }

    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cudnnbackendgetattribute_1_svc(ptr descriptor, int attributeName, int attributeType, int64_t requestedElementCount, mem_result *result, struct svc_req *rqstp)
{
    void *arrayOfElements = NULL;
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    if (attributeType < 0 || attributeType >= CUDNN_TYPE_RNG_DISTRIBUTION) {
        LOGE(LOG_ERROR, "attributeType out of range.");
        return 0;
    }
    result->mem_result_u.data.mem_data_len = sizeof(int64_t) + requestedElementCount*sizeof(backendAttributeSizes[attributeType]);
    if ((result->mem_result_u.data.mem_data_val = malloc(result->mem_result_u.data.mem_data_len)) == NULL) {
        LOGE(LOG_ERROR, "malloc failed");
        return 0;
    }
    if (requestedElementCount > 0) {
        void *data = result->mem_result_u.data.mem_data_val + sizeof(int64_t);
    }
    
    GSCHED_RETAIN;
    result->err = cudnnBackendGetAttribute(
        (cudnnBackendDescriptor_t)resource_mg_get_or_null(&rm_cudnn_backendds, (void*)descriptor),
        (cudnnBackendAttributeName_t)attributeName,
        (cudnnBackendAttributeType_t)attributeType,
        requestedElementCount,
        (int64_t*)result->mem_result_u.data.mem_data_val,
        arrayOfElements);
    
    LOGE(LOG_DEBUG, "desc: %p, name: %d, type: %d, requestedElementCount: %zd, elementCount: %zd, arrayOfElements: %p -> %d", descriptor, attributeName, attributeType, requestedElementCount, *result->mem_result_u.data.mem_data_val, arrayOfElements, result->err);

    GSCHED_RELEASE;
    return 1;
}
bool_t rpc_cudnnbackendexecute_1_svc(ptr handle, ptr executionPlan, ptr variantPack, int *result, struct svc_req *rqstp)
{
    RECORD_API(rpc_cudnnbackendexecute_1_argument);
    RECORD_NARG(handle);
    RECORD_NARG(executionPlan);
    RECORD_NARG(variantPack);
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);

    GSCHED_RETAIN;
    *result = cudnnBackendExecute(
        (cudnnHandle_t)resource_mg_get_or_null(&rm_cudnn, (void*)handle),
        (cudnnBackendDescriptor_t)resource_mg_get_or_null(&rm_cudnn_backendds, (void*)executionPlan),
        (cudnnBackendDescriptor_t)resource_mg_get_or_null(&rm_cudnn_backendds, (void*)variantPack));
    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}
