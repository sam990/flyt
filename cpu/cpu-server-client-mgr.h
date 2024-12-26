/**
 * FILE: cpu-server-client-mgr.h
 * -----------------------------
 * Exclusively linked to the flyt server
 * backend.
 */

# ifndef __FLYT_CPU_SERVER_CLIENT_MGR_H
# define __FLYT_CPU_SERVER_CLIENT_MGR_H

#include <stdio.h>
#include <cudnn.h>
#include "resource-mg.h"
#include "resource-map.h"

#define INIT_MEM_SLOTS 4096
#define INIT_STREAM_SLOTS 16
#define INIT_MODULE_SLOTS 8
#define INIT_FUNCTION_SLOTS 128
#define INIT_VAR_SLOTS 128
#define INIT_EVENT_SLOTS 16
#define INIT_CUDNN_SLOTS 16
#define INIT_CUDNN_TENSOR_SLOTS 128
#define INIT_CUDNN_FILTER_SLOTS 128
#define INIT_CUDNN_POOLING_SLOTS 128
#define INIT_CUDNN_ACTIVATION_SLOTS 128
#define INIT_CUDNN_LRN_SLOTS 128
#define INIT_CUDNN_CONV_SLOTS 128
#define INIT_CUDNN_BACKEND_SLOTS 16


enum MODULE_LOAD_TYPE {
    MODULE_LOAD_DATA = 0,
    MODULE_LOAD = 1,
};

enum MODULE_GET_FUNCTION_TYPE {
    MODULE_GET_FUNCTION = 0
};

enum MODULE_GET_GLOBAL_TYPE {
    MODULE_GET_GLOBAL = 0
};

enum MEM_ALLOC_TYPE {
    MEM_ALLOC_TYPE_DEFAULT = 0,
    MEM_ALLOC_TYPE_3D = 1,
    MEM_ALLOC_TYPE_PITCH = 2,
};

enum CUDNN_TENSOR_DESC_TYPE {
    CUDNN_TENSOR_DESC_TYPE_DEFAULT = 0,
    CUDNN_TENSOR_DESC_TYPE_4D = 1,
    CUDNN_TENSOR_DESC_TYPE_4D_EX = 2,
    CUDNN_TENSOR_DESC_TYPE_ND = 3,
    CUDNN_TENSOR_DESC_TYPE_ND_EX = 4,
};

enum CUDNN_FILTER_DESC_TYPE {
    CUDNN_FILTER_DESC_TYPE_DEFAULT = 0,
    CUDNN_FILTER_DESC_TYPE_4D = 1,
    CUDNN_FILTER_DESC_TYPE_ND = 2,
};

enum CUDNN_POOLING_DESC_TYPE {
    CUDNN_POOLING_DESC_TYPE_DEFAULT = 0,
    CUDNN_POOLING_DESC_TYPE_2D = 0,
    CUDNN_POOLING_DESC_TYPE_ND = 1,
};

enum CUDNN_CONV_DESC_TYPE {
    CUDNN_CONV_DESC_TYPE_DEFAULT = 0,
    CUDNN_CONV_DESC_TYPE_2D = 0,
    CUDNN_CONV_DESC_TYPE_ND = 1,
};

typedef struct __mem_alloc_args {
    enum MEM_ALLOC_TYPE type;
    long long size;
    size_t depth;
    size_t height;
    size_t width;
    size_t pitch;
    long long arg6; 
    size_t padded_size;
    size_t idx;
} mem_alloc_args_t;


typedef struct __cudnn_tensor_desc_args {
    enum CUDNN_TENSOR_DESC_TYPE type;
    int dataType;
    int nbDims;
    int dims[CUDNN_DIM_MAX];
    int strides[CUDNN_DIM_MAX];
    int format;
} cudnn_tensor_desc_args_t;

typedef struct __cudnn_filter_desc_args {
    enum CUDNN_FILTER_DESC_TYPE type;
    int dataType;
    int nbDims;
    int dims[CUDNN_DIM_MAX];
    int format;
} cudnn_filter_desc_args_t;

typedef struct __cudnn_pooling_desc_args {
    enum CUDNN_POOLING_DESC_TYPE type;
    int mode;
    int maxpoolingNanOpt;
    int nbDims;
    int windowDimA[CUDNN_DIM_MAX-2];
    int paddingA[CUDNN_DIM_MAX-2];
    int strideA[CUDNN_DIM_MAX-2];
} cudnn_pooling_desc_args_t;

typedef struct __cudnn_activation_desc_args {
    int activateSet;
    int mode;
    int reluNanOpt;
    double coef;
    int swishBetaSet;
    double swishBeta;
} cudnn_activation_desc_args_t;

typedef struct __cudnn_lrn_desc_args {
    unsigned int lrnN;
    double lrnAlpha;
    double lrnBeta;
    double lrnK;
} cudnn_lrn_desc_args_t;

typedef struct __cudnn_conv_desc_args {
    enum CUDNN_CONV_DESC_TYPE type;
    int dataType;
    int nbDims;
    int padA[CUDNN_DIM_MAX];
    int filterStrideA[CUDNN_DIM_MAX];
    int dilationA[CUDNN_DIM_MAX];
    int mode;
    int mathType;
    int mathTypeSet;
    int groupCount;
    int groupCountSet;
} cudnn_conv_desc_args_t;

typedef struct __cudnn_backend_attr {
    int attributeName;
    int attributeType;
    int64_t elementCount;
    void* arrayOfElements;
} cudnn_backend_attr_t;

typedef struct __cudnn_backend_desc_args {
    int descriptorType;
    size_t numAttrs;
    int initialized;
    int finalized;
    list attributes;
} cudnn_backend_desc_args_t;

typedef struct __var_register_args {
    void* module;
    char* deviceName;
    size_t size;
    void* data;
} var_register_args_t;


enum STREAM_CREATE_TYPE {
    STREAM_CREATE_TYPE_DEFAULT = 0,
    STREAM_CREATE_TYPE_FLAGS = 1,
    STREAM_CREATE_TYPE_PRIORITY = 2,
};

typedef struct __stream_create_args {
    enum STREAM_CREATE_TYPE type;
    int flags;
    int priority;
} stream_create_args_t;


typedef struct __addr_data_pair {
    void* addr;
    struct {
        int type;
        size_t size;
        void* data;
    } reg_data;
} addr_data_pair_t;

typedef struct __event_args {
    int flags;
    bool_t time_recorded;
} event_args_t;

typedef struct __cricket_client {
    int pid;
    resource_mg gpu_mem;
    void* default_stream;
    resource_map* custom_streams;
    resource_mg modules;
    resource_mg functions;
    resource_mg vars;
    resource_map* events;
    size_t malloc_idx;
    // further can be added

    // cudnn related resources
    resource_map *rm_cudnn;
    resource_map *rm_cudnn_tensors;
    resource_map *rm_cudnn_tensor_transforms;
    resource_map *rm_cudnn_filters;
    resource_map *rm_cudnn_poolings;
    resource_map *rm_cudnn_activations;
    resource_map *rm_cudnn_lrns;
    resource_map *rm_cudnn_convs;
    resource_map *rm_cudnn_backends;
} cricket_client;

typedef uint64_t cricket_client_iter;


int init_cpu_server_client_mgr();

void free_cpu_server_client_mgr();

cricket_client *create_client(int pid);

int add_new_client(int pid, int xp_fd);

int add_restored_client(cricket_client *client);

int move_restored_client(int pid, int xp_fd);

cricket_client* get_client(int xp_fd);

cricket_client* get_client_by_pid(int pid);

int remove_client_ptr(cricket_client *client);

int remove_client(int xp_fd);

cricket_client_iter get_client_iter();

cricket_client *get_next_client(cricket_client_iter *iter);

cricket_client *get_next_restored_client(cricket_client_iter *iter);

void free_variable_data(addr_data_pair_t *pair);

void free_function_data(addr_data_pair_t *pair);

void free_module_data(addr_data_pair_t *pair);

int free_client_stream(cricket_client *client, uint64_t stream);

int free_client_cudnn_handle(cricket_client *client, uint64_t handle);

int free_client_cudnn_tensor(cricket_client *client, uint64_t tensor);

int free_client_cudnn_filter(cricket_client *client, uint64_t filter);

int free_client_cudnn_pooling(cricket_client *client, uint64_t pooling);

int free_client_cudnn_activation(cricket_client *client, uint64_t activation);

int free_client_cudnn_lrn(cricket_client *client, uint64_t lrn);

int free_client_cudnn_conv(cricket_client *client, uint64_t conv);

int free_client_cudnn_backend(cricket_client *client, uint64_t backend);

int fetch_variable_data_to_host(void);

int dump_module_data(resource_mg_map_elem *elem, FILE *fp);

int load_module_data(resource_mg_map_elem *elem, FILE *fp);

int dump_variable_data(resource_mg_map_elem *elem, FILE *fp);

int load_variable_data(resource_mg_map_elem *elem, FILE *fp);

int dump_function_data(resource_mg_map_elem *elem, FILE *fp);

int load_function_data(resource_mg_map_elem *elem, FILE *fp);

int remove_client_by_pid(int pid);

int dealloc_client_resources();

int map_cudnn_backend_attribute(cricket_client* client, int attributeType, int64_t elemCount, void *data, void *dest);


#define GET_CLIENT(result) cricket_client *client = get_client(rqstp->rq_xprt->xp_fd); \
    if (client == NULL) { \
        LOGE(LOG_ERROR, "error getting client"); \
        result= cudaErrorInvalidValue; \
        GSCHED_RELEASE; \
        return 1; \
    }

#define GET_CLIENT_CUDNN(result) cricket_client *client = get_client(rqstp->rq_xprt->xp_fd); \
    if (client == NULL) { \
        LOGE(LOG_ERROR, "error getting client"); \
        result= CUDNN_STATUS_BAD_PARAM; \
        GSCHED_RELEASE; \
        return 1; \
    }


#define GET_STREAM(stream_ptr, stream, result) \
    if (stream == 0) { \
        stream_ptr = client->default_stream; \
    } else if (!resource_map_contains(client->custom_streams, (void*)stream)) { \
        LOGE(LOG_ERROR, "stream not found in custom streams"); \
        result = cudaErrorInvalidValue; \
        GSCHED_RELEASE; \
        return 1; \
    } else { \
        stream_ptr = resource_map_get_addr(client->custom_streams, (void*)stream); \
    }

#define GET_CUDNN_HANDLE(client_addr) \
    (cudnnHandle_t)resource_map_get_addr(client->rm_cudnn, (void*)client_addr)

#define GET_CUDNN_TENSOR(client_addr) \
    (cudnnTensorDescriptor_t)resource_map_get_addr(client->rm_cudnn_tensors, (void*)client_addr)

#define GET_CUDNN_FILTER(client_addr) \
    (cudnnFilterDescriptor_t)resource_map_get_addr(client->rm_cudnn_filters, (void*)client_addr)

#define GET_CUDNN_TENSOR_TRANSFORM(client_addr) \
    (cudnnTensorTransformDescriptor_t)resource_map_get_addr(client->rm_cudnn_tensor_transforms, (void*)client_addr)

#define GET_CUDNN_POOLING(client_addr) \
    (cudnnPoolingDescriptor_t)resource_map_get_addr(client->rm_cudnn_poolings, (void*)client_addr)

#define GET_CUDNN_ACTIVATION(client_addr) \
    (cudnnActivationDescriptor_t)resource_map_get_addr(client->rm_cudnn_activations, (void*)client_addr)

#define GET_CUDNN_LRN(client_addr) \
    (cudnnLRNDescriptor_t)resource_map_get_addr(client->rm_cudnn_lrns, (void*)client_addr)

#define GET_CUDNN_CONV(client_addr) \
    (cudnnConvolutionDescriptor_t)resource_map_get_addr(client->rm_cudnn_convs, (void*)client_addr)

#define GET_CUDNN_BACKEND(client_addr) \
    (cudnnBackendDescriptor_t)resource_map_get_addr(client->rm_cudnn_backends, (void*)client_addr)

// #define GET_SPL_MEMORY(mem_ptr, client_addr, result) \
//     if (!resource_map_contains(client->gpu_mem, (void*)client_addr)) { \
//         LOGE(LOG_ERROR, "memory not found in gpu_mem"); \
//         result = cudaErrorInvalidValue; \
//         GSCHED_RELEASE; \
//         return 1; \
//     } \
//     mem_ptr = resource_map_get_addr(client->gpu_mem, (void*)client_addr);

#define GET_VARIABLE(var_ptr, client_addr, result) \
    if ((var_ptr = resource_mg_get_or_null(&client->vars, (void *)client_addr)) == NULL) { \
        LOGE(LOG_ERROR, "variable not found in gpu_vars"); \
        result = cudaErrorInvalidValue; \
        GSCHED_RELEASE; \
        return 1; \
    }


#endif // __FLYT_CPU_SERVER_CLIENT_MGR_H
