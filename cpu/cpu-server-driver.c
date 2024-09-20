#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

#include "cpu_rpc_prot.h"
#include "cpu-server-driver-hidden.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "resource-mg.h"
#define WITH_RECORDER
#include "api-recorder.h"
#include "gsched.h"
#include "cpu-server-resource-controller.h"
#include "cpu-server-client-mgr.h"

#define GET_CLIENT_DRV(err) \
    cricket_client *client = get_client(rqstp->rq_xprt->xp_fd); \
    if (client == NULL) { \
        LOGE(LOG_ERROR, "client not found"); \
        err = CUDA_ERROR_NOT_INITIALIZED; \
        GSCHED_RELEASE; \
        return 1; \
    }


int server_driver_init(int restore)
{
    #ifdef WITH_IB
    #endif //WITH_IB
   
    return 0;
}

#include <cuda_runtime_api.h>
#include "cpu-server-driver.h"

// restore elf_modules
int server_driver_modules_restore(resource_mg *modules)
{

    LOGE(LOG_DEBUG, "Restoring modules");

    if (modules == NULL) {
        LOGE(LOG_ERROR, "modules is NULL");
        return 1;
    }

    pthread_mutex_lock(&modules->mutex);
    size_t num_elfs = modules->map_res.length;

    LOGE(LOG_DEBUG, "Restoring %zu modules", num_elfs);

    for (size_t i = 0; i < num_elfs; ++i) {
        resource_mg_map_elem *elem = ((resource_mg_map_elem*)list_get(&modules->map_res, i));
        if (elem == NULL) {
            LOGE(LOG_ERROR, "elem is NULL");
    	    pthread_mutex_unlock(&modules->mutex);
            return 1;
        }

        addr_data_pair_t *map = (addr_data_pair_t *)elem->cuda_address;
        
        LOGE(LOG_DEBUG, "module_info %p", map);

        if (map == NULL) {
            LOGE(LOG_ERROR, "map is NULL");
    	    pthread_mutex_unlock(&modules->mutex);
            return 1;
        }

        CUmodule module = NULL;
        CUresult res;

        switch (map->reg_data.type) {
            case MODULE_LOAD_DATA:
                mem_data *elf_args = map->reg_data.data;
                if (elf_args == NULL || elf_args->mem_data_val == NULL) {
                    LOGE(LOG_ERROR, "elf_args is NULL");
		    pthread_mutex_unlock(&modules->mutex);
                    return 1;
                }

                LOGE(LOG_DEBUG, "Restoring module %p, len %u", elf_args->mem_data_val, elf_args->mem_data_len);

                if ((res = cuModuleLoadData(&module, elf_args->mem_data_val)) != CUDA_SUCCESS) {
                    LOGE(LOG_ERROR, "cuModuleLoadData failed: %d", res);
		    pthread_mutex_unlock(&modules->mutex);
                    return 1;
                }
                break;
            case MODULE_LOAD:
                char *path = map->reg_data.data;
                if (path == NULL) {
                    LOGE(LOG_ERROR, "path is NULL");
		    pthread_mutex_unlock(&modules->mutex);
                    return 1;
                }

                LOGE(LOG_DEBUG, "Restoring module %s", path);

                if ((res = cuModuleLoad(&module, path)) != CUDA_SUCCESS) {
                    LOGE(LOG_ERROR, "cuModuleLoad failed: %d", res);
		    pthread_mutex_unlock(&modules->mutex);
                    return 1;
                }
                break;
            default:
                LOGE(LOG_ERROR, "Unknown module type: %d", map->reg_data.type);
		pthread_mutex_unlock(&modules->mutex);
                return 1;

        }    
        map->addr = module;
    }
    pthread_mutex_unlock(&modules->mutex);

    LOGE(LOG_DEBUG, "Restored %zu modules", num_elfs);
    

    return 0;
}

int server_driver_function_restore(resource_mg *func, resource_mg *modules) 
{
    LOGE(LOG_DEBUG, "Restoring functions");
    pthread_mutex_lock(&func->mutex);
    size_t num_funcs = func->map_res.length;
    for (size_t i = 0; i < num_funcs; i++) {
        addr_data_pair_t *map = ((resource_mg_map_elem*)list_get(&func->map_res, i))->cuda_address;

        CUfunction func_d_ptr;

	if(map == NULL) {
                LOGE(LOG_DEBUG, "map is null");
		continue;
	}

        switch (map->reg_data.type) {
            case MODULE_GET_FUNCTION:
                rpc_cumodulegetfunction_1_argument *arg = map->reg_data.data;
                ptr fatCubinHandle = arg->arg1;
                char *deviceName = arg->arg2;

                LOGE(LOG_DEBUG, "rpc_cumodulegetfunction(fatCubinHandle: %p, deviceName: %s)", fatCubinHandle, deviceName);

                addr_data_pair_t *module = NULL;
                CUresult res;

                if (resource_mg_get(modules, (void*)fatCubinHandle, (void *)&module) != 0) {
                    LOGE(LOG_ERROR, "%p not found in resource manager - we cannot call a function from an unknown module.", fatCubinHandle);
    		    pthread_mutex_unlock(&func->mutex);
                    return 1;
                }
                
                if ((res = cuModuleGetFunction(&func_d_ptr,
                                module->addr,
                                deviceName)) != CUDA_SUCCESS) {
                    LOGE(LOG_ERROR, "cuModuleGetFunction failed: %d", res);
    		    pthread_mutex_unlock(&func->mutex);
                    return 1;
                }     
                break;
            default:
                LOGE(LOG_ERROR, "Unknown function type: %d", map->reg_data.type);
    		pthread_mutex_unlock(&func->mutex);
                return 1;
        }

        map->addr = func_d_ptr;
    }
    pthread_mutex_unlock(&func->mutex);

    LOGE(LOG_DEBUG, "Restored %zu functions", num_funcs);

    return 0;
}

int server_driver_var_restore(resource_mg *vars, resource_mg *modules)
{
    LOGE(LOG_DEBUG, "Restoring vars");
    size_t num_vars = vars->map_res.length;

    for (size_t i = 0; i < num_vars; ++i) {
        addr_data_pair_t *map = ((resource_mg_map_elem*)list_get(&vars->map_res, i))->cuda_address;
	if(map == NULL) {
                LOGE(LOG_DEBUG, "map is null");
		continue;
	}
        CUdeviceptr dptr;
        size_t dsize; 
        switch (map->reg_data.type) {
            case MODULE_GET_GLOBAL:
                var_register_args_t *arg = map->reg_data.data;

                addr_data_pair_t *module = NULL;
                CUresult res;
                if (resource_mg_get(modules, (void*)arg->module, (void *)&module) != 0) {
                    LOGE(LOG_ERROR, "%p not found in resource manager - we cannot call a function from an unknown module.", arg->module);
                    return 1;
                }

                if ((res = cuModuleGetGlobal(&dptr, &dsize, module->addr, arg->deviceName)) != CUDA_SUCCESS) {
                    LOGE(LOG_ERROR, "cuModuleGetGlobal failed: %d", res);
                    return 1;
                }

                // restore data
                if (arg->data != NULL) {
                    cudaMemcpyToSymbol((void *)dptr, arg->data, arg->size, 0, cudaMemcpyHostToDevice);
                }

                break;
            default:
                LOGE(LOG_ERROR, "Unknown var type: %d", map->reg_data.type);
                return 1;
        }

        map->addr = (void *)dptr;
    }

    LOGE(LOG_DEBUG, "Restored %zu vars", num_vars);
    return 0;
}

// restore the checkpointed client state on the server.
static int __server_driver_ctx_state_restore(int ckp_restore) {

    LOGE(LOG_DEBUG, "Restoring context state");

    cricket_client_iter iter = get_client_iter();

    cricket_client * (*iter_fn) (cricket_client_iter *) = ckp_restore ? get_next_restored_client : get_next_client;

    cricket_client *client;

    while ((client = iter_fn(&iter)) != NULL ) {


        LOGE(LOG_DEBUG, "Restoring client %p. pid: %d", client, client->pid);
        
        int ret = server_driver_modules_restore(&client->modules) || 
        server_driver_function_restore(&client->functions, &client->modules) ||
        server_driver_var_restore(&client->vars, &client->modules);

        if (ret != 0) {
            LOGE(LOG_ERROR, "restoring client failed");
            return 1;
        }
        cudaError_t err;
        err = cudaStreamCreate((cudaStream_t*)&client->default_stream);
        if (err != cudaSuccess) {
            LOGE(LOG_ERROR, "cudaStreamCreate failed: %s", cudaGetErrorString(err));
            return 1;
        }

        resource_map_iter *stream_iter = resource_map_init_iter(client->custom_streams);
        if (stream_iter == NULL) {
            LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map iterator");
            return 1;
        }

        uint64_t stream_idx;
        while ((stream_idx = resource_map_iter_next(stream_iter)) != 0) {
            cudaStream_t newStream;
            err = cudaStreamCreate(&newStream);
            // TODO: include custom cuda stream flags
            if (err != cudaSuccess) {
                LOGE(LOG_ERROR, "cudaStreamCreate failed: %s", cudaGetErrorString(err));
                resource_map_free_iter(stream_iter);
                return 1;
            }
            if (resource_map_update_addr_idx(client->custom_streams, stream_idx, newStream) != 0) {
                LOGE(LOG_ERROR, "resource_map_update failed, stream_idx: %lu", stream_idx);
                resource_map_free_iter(stream_iter);
                return 1;
            }
        }

        resource_map_free_iter(stream_iter);
    }

    LOGE(LOG_DEBUG, "Context state restored");

    return 0;
}

int server_driver_reload_modules_data(cricket_client *client) {
    
    if (server_driver_modules_restore(&client->modules) != 0) {
        LOGE(LOG_ERROR, "server_driver_modules_restore failed");
        return 1;
    }

    if (server_driver_function_restore(&client->functions, &client->modules) != 0) {
        LOGE(LOG_ERROR, "server_driver_function_restore failed");
        return 1;
    }

    if (server_driver_var_restore(&client->vars, &client->modules) != 0) {
        LOGE(LOG_ERROR, "server_driver_var_restore failed");
        return 1;
    }
    
}

int server_driver_ctx_state_restore(CUcontext ctx) {

    cuCtxPushCurrent(ctx);
    int ret = fetch_variable_data_to_host();
    cuCtxPopCurrent(&ctx);

    if (ret != 0) {
        LOGE(LOG_ERROR, "fetch_variable_data_to_host failed");
        return 1;
    }

    return __server_driver_ctx_state_restore(0);
}

int server_driver_ctx_state_restore_ckp(void) {
    return __server_driver_ctx_state_restore(1);
}


// Does not support checkpoint/restart yet
bool_t rpc_elf_load_1_svc(mem_data elf, ptr module_key, int *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    LOGE(LOG_DEBUG, "rpc_elf_load(elf: %p, len: %#x, module_key: %p)", elf.mem_data_val, elf.mem_data_len, module_key);
    CUresult res;
    CUmodule module = NULL;

    GET_CLIENT_DRV(*result);

    LOGE(LOG_DEBUG, "client: %p, client pid: %d", client, client->pid);
    
    if ((res = cuModuleLoadData(&module, elf.mem_data_val)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cuModuleLoadData failed: %d", res);
        *result = res;
        GSCHED_RELEASE;
        return 1;
    }

    // We add our module using module_key as key. This means a fatbinaryHandle on the client is translated
    // to a CUmodule on the server.

    mem_data *elf_args = malloc(sizeof(mem_data));
    elf_args->mem_data_len = elf.mem_data_len;
    elf_args->mem_data_val = malloc(elf.mem_data_len);
    memcpy(elf_args->mem_data_val, elf.mem_data_val, elf.mem_data_len);

    addr_data_pair_t *module_info = malloc(sizeof(addr_data_pair_t));
    module_info->addr = module;
    module_info->reg_data.type = MODULE_LOAD_DATA;
    module_info->reg_data.size = sizeof(mem_data);
    module_info->reg_data.data = elf_args;

    LOGE(LOG_DEBUG, "module: %p, module_info: %p, elfargs: %p", module, module_info, elf_args);

    if (resource_mg_add_sorted(&client->modules, (void*)module_key, (void*)module_info) != 0) {
        LOGE(LOG_ERROR, "resource_mg_create failed");

        free(elf_args->mem_data_val);
        free(elf_args);
        free(module_info);

        *result = -1;
        GSCHED_RELEASE;
        return 1;
    }

    *result = 0;
    GSCHED_RELEASE;
    return 1;
}

// Does not support checkpoint/restart yet
// TODO: We should also remove associated function handles
bool_t rpc_elf_unload_1_svc(ptr elf_handle, int *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    LOGE(LOG_DEBUG, "rpc_elf_unload(elf_handle: %p)", elf_handle);
    addr_data_pair_t* module = NULL;
    CUresult res;

    GET_CLIENT_DRV(*result);
    
    
    if (resource_mg_get(&client->modules, (void*)elf_handle, (void *)&module) != 0) {
        LOG(LOG_ERROR, "resource_mg_get failed");
        *result = -1;
        GSCHED_RELEASE;
        return 1;
    }

    LOGE(LOG_DEBUG,"module: %p", module);

    // if ((res = resource_mg_remove(&rm_modules, (void*)elf_handle)) != CUDA_SUCCESS) {
    //     LOG(LOG_ERROR, "resource_mg_create failed: %d", res);
    //     result->err = res;
    //     return 1;
    // }

    if ((res = cuModuleUnload(module->addr)) != CUDA_SUCCESS) {
        const char *errstr;
        cuGetErrorString(res, &errstr);
        LOG(LOG_ERROR, "cuModuleUnload failed: %s (%d)", errstr, res);
        *result = res;
        GSCHED_RELEASE;
        return 1;
    }

    free(((mem_data*)module->reg_data.data)->mem_data_val);
    free(module->reg_data.data);
    free(module);

    resource_mg_remove(&client->modules, (void*)elf_handle);
   
    *result = 0;
    GSCHED_RELEASE;
    return 1;
}

// Does not support checkpoint/restart yet
bool_t rpc_register_function_1_svc(ptr fatCubinHandle, ptr hostFun, char* deviceFun,
                            char* deviceName, int thread_limit, ptr_result *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    addr_data_pair_t *module = NULL;
    RECORD_API_STR(rpc_register_function_1_argument, 2);
    RECORD_ARG(1, fatCubinHandle);
    RECORD_ARG(2, hostFun);
    RECORD_ARG_STR(3, deviceFun);
    RECORD_ARG_STR(4, deviceName);
    RECORD_ARG(5, thread_limit);
    LOG(LOG_DEBUG, "rpc_register_function(fatCubinHandle: %p, hostFun: %p, deviceFun: %s, deviceName: %s, thread_limit: %d)",
        fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit);
    
    GET_CLIENT_DRV(result->err);

    //resource_mg_print(&rm_modules);
    if (resource_mg_get(&client->modules, (void*)fatCubinHandle, (void *)&module) != 0) {
        LOGE(LOG_ERROR, "%p not found in resource manager - we cannot call a function from an unknown module.", fatCubinHandle);
        result->err = -1;
        GSCHED_RELEASE;
        return 1;
    }

    CUfunction func_d_ptr;

    result->err = cuModuleGetFunction(&func_d_ptr,
                    module->addr,
                    deviceName);
    

    if (result->err == CUDA_SUCCESS) {

        rpc_cumodulegetfunction_1_argument *reg_args = malloc(sizeof(rpc_cumodulegetfunction_1_argument));
        // list_append(&rm_function_regs, (void**)&reg_args);
        reg_args->arg1 = fatCubinHandle;
        reg_args->arg2 = strdup(deviceName);

        addr_data_pair_t *func_info = malloc(sizeof(addr_data_pair_t));
        func_info->addr = func_d_ptr;
        func_info->reg_data.type = MODULE_GET_FUNCTION;
        func_info->reg_data.size = sizeof(rpc_cumodulegetfunction_1_argument);
        func_info->reg_data.data = reg_args;

        resource_mg_add_sorted(&client->functions, (void*)hostFun, (void*)func_info);
    }

    RECORD_RESULT(ptr_result_u, *result);
    GSCHED_RELEASE;
    return 1;
}

// Does not support checkpoint/restart yet
bool_t rpc_register_var_1_svc(ptr fatCubinHandle, ptr hostVar, ptr deviceAddress, char *deviceName, int ext, size_t size,
                        int constant, int global, int *result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    RECORD_API(rpc_register_var_1_argument);
    RECORD_ARG(1, fatCubinHandle);
    RECORD_ARG(2, hostVar);
    RECORD_ARG(3, deviceAddress);
    RECORD_ARG(4, deviceName);
    RECORD_ARG(5, ext);
    RECORD_ARG(6, size);
    RECORD_ARG(7, constant);
    RECORD_ARG(8, global);
    
    LOG(LOG_DEBUG, "rpc_register_var(fatCubinHandle: %p, hostVar: %p, deviceAddress: %p, deviceName: %s, "
                   "ext: %d, size: %d, constant: %d, global: %d)",
                   fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);

    GET_CLIENT_DRV(*result);

    
    CUdeviceptr dptr = 0;
    size_t d_size = 0;
    CUresult res;
    addr_data_pair_t *module = NULL;

    if (resource_mg_get(&client->modules, (void*)fatCubinHandle, (void *)&module) != 0) {
        LOGE(LOG_ERROR, "%p not found in resource manager - we cannot call a function from an unknown module.", fatCubinHandle);
        *result = -1;
        GSCHED_RELEASE;
        return 1;
    }
    if ((res = cuModuleGetGlobal(&dptr, &d_size, module->addr, deviceName)) != CUDA_SUCCESS) {
        LOGE(LOG_ERROR, "cuModuleGetGlobal failed: %d", res);
        *result = 1;
        GSCHED_RELEASE;
        return 1;
    }


    var_register_args_t *reg_args = malloc(sizeof(var_register_args_t));

    reg_args->module = (void*)fatCubinHandle;
    reg_args->deviceName = strdup(deviceName);
    reg_args->size = d_size;
    reg_args->data = NULL;


    addr_data_pair_t *var_info = malloc(sizeof(addr_data_pair_t));
    var_info->addr = (void *)dptr;
    var_info->reg_data.type = MODULE_GET_GLOBAL;
    var_info->reg_data.size = sizeof(var_register_args_t);
    var_info->reg_data.data = (void *)reg_args;

    if (resource_mg_add_sorted(&client->vars, (void*)hostVar, (void*)var_info) != 0) {
        LOGE(LOG_ERROR, "error in resource manager");
        free(reg_args->deviceName);
        free(reg_args);
        free(var_info);
        *result = 1;
    } else {
        *result = 0;
    }
    RECORD_RESULT(integer, *result);
    GSCHED_RELEASE;
    return 1;
}

int server_driver_deinit(void)
{
    return 0;
}

bool_t rpc_cudevicegetcount_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    result->err = cuDeviceGetCount(&result->int_result_u.data);
    return 1;
}

bool_t rpc_cuinit_1_svc(int argp, int *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    *result = cuInit(argp);
    return 1;
}

bool_t rpc_cudrivergetversion_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    result->err = cuDriverGetVersion(&result->int_result_u.data);
    return 1;
}

bool_t rpc_cudeviceget_1_svc(int ordinal, int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGet(&result->int_result_u.data, ordinal);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetname_1_svc(int dev, str_result *result, struct svc_req *rqstp)
{
    result->str_result_u.str = malloc(128);
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetName(result->str_result_u.str, 128, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicetotalmem_1_svc(int dev, u64_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceTotalMem(&result->u64_result_u.u64, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetattribute_1_svc(int attribute, int dev, int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetAttribute(&result->int_result_u.data, attribute, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetuuid_1_svc(int dev, str_result *result, struct svc_req *rqstp)
{
    CUuuid uuid;
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetUuid(&uuid, dev);
    if (result->err == 0) {
        memcpy(result->str_result_u.str, uuid.bytes, 16);
    }
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cuctxgetcurrent_1_svc(ptr_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuCtxGetCurrent((struct CUctx_st**)&result->ptr_result_u.ptr);
    if ((void*)result->ptr_result_u.ptr != NULL) {
        unsigned int version = 0;
        cuCtxGetApiVersion((CUcontext)result->ptr_result_u.ptr, &version);
        LOG(LOG_DEBUG, "ctxapi version: %d", version);
    }
    GSCHED_RELEASE;
    return 1;
}

//TODO: Calling this might break things within the scheduler.
bool_t rpc_cuctxsetcurrent_1_svc(uint64_t ptr, int *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    *result = cuCtxSetCurrent((struct CUctx_st*)ptr);
    return 1;
}

//TODO: Calling this might break things within the scheduler.
bool_t rpc_cudeviceprimaryctxretain_1_svc(int dev, ptr_result *result,
                                          struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    result->err = cuDevicePrimaryCtxRetain((struct CUctx_st**)&result->ptr_result_u.ptr, dev);
    return 1;
}

bool_t rpc_cumodulegetfunction_1_svc(uint64_t module, char *name, ptr_result *result,
                                     struct svc_req *rqstp)
{
    RECORD_API(rpc_cumodulegetfunction_1_argument);
    RECORD_ARG(1, module);
    RECORD_ARG(2, name);
    LOG(LOG_DEBUG, "(fd:%d) %s(%s)", rqstp->rq_xprt->xp_fd, __FUNCTION__, name);
    GSCHED_RETAIN;

    GET_CLIENT_DRV(result->err);

    addr_data_pair_t *module_ptr = NULL;
    if (resource_mg_get(&client->modules, (void*)module, (void *)&module_ptr) != 0) {
        LOGE(LOG_ERROR, "module not found in resource manager");
        result->err = CUDA_ERROR_INVALID_HANDLE;
        GSCHED_RELEASE;
        return 1;
    }

    result->err = cuModuleGetFunction((CUfunction*)&result->ptr_result_u.ptr,
                    (CUmodule)module_ptr,
                    name);

    if (result->err == 0) {
        rpc_cumodulegetfunction_1_argument *reg_args = malloc(sizeof(rpc_cumodulegetfunction_1_argument));
        reg_args->arg1 = module;
        reg_args->arg2 = strdup(name);

        addr_data_pair_t *func_info = malloc(sizeof(addr_data_pair_t));
        func_info->addr = (void*)result->ptr_result_u.ptr;
        func_info->reg_data.type = MODULE_GET_FUNCTION;
        func_info->reg_data.size = sizeof(rpc_cumodulegetfunction_1_argument);
        func_info->reg_data.data = reg_args;

        if (resource_mg_add_sorted(&client->functions, (void*)result->ptr_result_u.ptr, (void*)func_info) != 0) {
            LOGE(LOG_ERROR, "error in resource manager");
            free(reg_args->arg2);
            free(reg_args);
            free(func_info);
            result->err = CUDA_ERROR_UNKNOWN;
        }
    }
    
    RECORD_RESULT(ptr_result_u, *result);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cumoduleloaddata_1_svc(mem_data mem, ptr_result *result,
                                     struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    RECORD_API(mem_data);
    RECORD_SINGLE_ARG(mem);
    LOG(LOG_DEBUG, "%s(%p, %#0zx)", __FUNCTION__, mem.mem_data_val, mem.mem_data_len);

    GET_CLIENT_DRV(result->err);

    result->err = cuModuleLoadData((CUmodule*)&result->ptr_result_u.ptr, mem.mem_data_val);
    if (result->err != 0) {
        const char *err_str = NULL;
        cuGetErrorName(result->err, &err_str);
        LOGE(LOG_DEBUG, "cuModuleLoadData result: %s", err_str);
    }
    else {
        mem_data *elf_args = malloc(sizeof(mem_data));
        elf_args->mem_data_len = mem.mem_data_len;
        elf_args->mem_data_val = malloc(mem.mem_data_len);
        memcpy(elf_args->mem_data_val, mem.mem_data_val, mem.mem_data_len);
        
        addr_data_pair_t *module_info = malloc(sizeof(addr_data_pair_t));
        module_info->addr = (void*)result->ptr_result_u.ptr;
        module_info->reg_data.type = MODULE_LOAD_DATA;
        module_info->reg_data.data = elf_args;
        module_info->reg_data.size = sizeof(mem_data);

        if (resource_mg_add_sorted(&client->modules, (void*)result->ptr_result_u.ptr, (void*)module_info) != 0) {
            LOGE(LOG_ERROR, "resource_mg_create failed");
            free(elf_args->mem_data_val);
            free(elf_args);
            free(module_info);
            result->err = CUDA_ERROR_UNKNOWN;
        }
    }
    RECORD_RESULT(ptr_result_u, *result);
    GSCHED_RELEASE;
    return 1;
}
bool_t rpc_cumoduleload_1_svc(char* path, ptr_result *result,
                                     struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    RECORD_API(char*);
    RECORD_SINGLE_ARG(path);
    LOG(LOG_DEBUG, "%s(%s)", __FUNCTION__, path);

    GET_CLIENT_DRV(result->err);

    result->err = cuModuleLoad((CUmodule*)&result->ptr_result_u.ptr, path);
    if (result->err != 0) {
        const char *err_str = NULL;
        cuGetErrorName(result->err, &err_str);
        LOGE(LOG_DEBUG, "cuModuleLoad result: %s", err_str);
    }
    else {
        addr_data_pair_t *module_info = malloc(sizeof(addr_data_pair_t));
        module_info->addr = (void*)result->ptr_result_u.ptr;
        module_info->reg_data.type = MODULE_LOAD;
        module_info->reg_data.data = strdup(path);
        module_info->reg_data.size = strlen(path) + 1;

        if (resource_mg_add_sorted(&client->modules, (void*)result->ptr_result_u.ptr, (void*)module_info) != 0) {
            LOGE(LOG_ERROR, "resource_mg_create failed");
            cuModuleUnload((CUmodule)result->ptr_result_u.ptr);
            free(module_info->reg_data.data);
            free(module_info);
            result->err = CUDA_ERROR_UNKNOWN;
        }
    
    }
    RECORD_RESULT(ptr_result_u, *result);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cumoduleunload_1_svc(ptr module, int *result,
                                     struct svc_req *rqstp)
{
    RECORD_API(ptr);
    RECORD_SINGLE_ARG(module);
    LOG(LOG_DEBUG, "%s(%p)", __FUNCTION__, (void*)module);
    GSCHED_RETAIN;

    GET_CLIENT_DRV(*result);

    addr_data_pair_t *module_ptr = NULL;
    

    if (resource_mg_get(&client->modules, (void*)module, (void *)&module_ptr) != 0){
        LOGE(LOG_ERROR, "module not found in resource manager");
        *result = CUDA_ERROR_INVALID_HANDLE;
        GSCHED_RELEASE; 
        return 1;
    }

    *result = cuModuleUnload((CUmodule)module_ptr->addr);

    if (*result == CUDA_SUCCESS) {

        if (resource_mg_remove(&client->modules, (void*)module) != 0) {
            LOGE(LOG_ERROR, "resource_mg_remove failed");
            *result = CUDA_ERROR_UNKNOWN;
        }
        else {
            free_module_data(module_ptr);
        }
    }

    GSCHED_RELEASE;
    RECORD_RESULT(integer, *result);
    return 1;
}

bool_t rpc_cugeterrorstring_1_svc(int err, str_result *result,
                                     struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s", __FUNCTION__);
    const char* err_str = NULL;
    result->err = cuGetErrorString(err, &err_str);

    if ((result->str_result_u.str = malloc(128)) == NULL ||
        (strncpy(result->str_result_u.str, err_str, 128) == NULL)) {
        LOGE(LOG_ERROR, "error copying string");
    }

    return 1;
}

bool_t rpc_cudeviceprimaryctxgetstate_1_svc(int dev, dint_result *result,
                                      struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s(%d)", __FUNCTION__, dev);
    GSCHED_RETAIN;
    result->err = cuDevicePrimaryCtxGetState(dev, &(result->dint_result_u.data.i1),
                                            &(result->dint_result_u.data.i2));
    LOGE(LOG_DEBUG, "state: %d, flags: %d", result->dint_result_u.data.i1,
                                           result->dint_result_u.data.i2);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicegetproperties_1_svc(int dev, mem_result *result,
                                       struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s(%d)", __FUNCTION__, dev);
    GSCHED_RETAIN;
    if ((result->mem_result_u.data.mem_data_val = malloc(sizeof(CUdevprop))) == NULL) {
        result->err = CUDA_ERROR_OUT_OF_MEMORY;
    }
    result->mem_result_u.data.mem_data_len = sizeof(CUdevprop);
    result->err = cuDeviceGetProperties((CUdevprop*)result->mem_result_u.data.mem_data_val, dev);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cudevicecomputecapability_1_svc(int dev, dint_result *result,
                                           struct svc_req *rqstp)
{
    LOGE(LOG_DEBUG, "%s(%d)", __FUNCTION__, dev);
    GSCHED_RETAIN;
    result->err = cuDeviceComputeCapability(&(result->dint_result_u.data.i1),
                                            &(result->dint_result_u.data.i2),
                                            dev);
    GSCHED_RELEASE;
    return 1;
}

/*
bool_t rpc_cugetexporttable_1_svc(char *rpc_uuid, ptr_result *result,
                                  struct svc_req *rqstp)
{
    void *exportTable = NULL;
    size_t tablesize = 0;
    CUuuid uuid;
    LOG(LOG_DEBUG, printf("%s\n", __FUNCTION__);
    if (rpc_uuid == NULL)
        return 0;

    memcpy(uuid.bytes, rpc_uuid, 16);
    if ((result->err = cuGetExportTable((const void**)&exportTable,
                                        (const CUuuid*)&uuid) != 0)) {
        return 1;
    }
    if (((uint32_t*)exportTable)[1] > 0) {
        tablesize = 8;
        for (int i=1; i<8; ++i) {
            if (((void**)exportTable)[i] == NULL) {
                tablesize = i;
                break;
            }
        }
    } else {
        tablesize = *((uint64_t*)exportTable)/8;
    }
    printf("\ttablesize = %lu\n", tablesize);
    printf("\tpost %p->%p\n", exportTable, *(void**)exportTable);

    if (!(uint64_t)cd_svc_hidden_add_table(exportTable, tablesize)) {
        fprintf(stderr, "\tfailed to add table!\n");
        return 0;
    }
    result->ptr_result_u.ptr = (uint64_t)exportTable;

    return 1;
}*/

bool_t rpc_cumemalloc_1_svc(uint64_t size, ptr_result *result,
                                     struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuMemAlloc_v2((CUdeviceptr*)&result->ptr_result_u.ptr, (size_t)size);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cuctxgetdevice_1_svc(int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuCtxGetDevice((CUdevice*)&result->int_result_u.data);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_cumemcpyhtod_1_svc(uint64_t dptr, mem_data hptr, int *result,
                                     struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s(%p,%p,%d)", __FUNCTION__, dptr, hptr.mem_data_val, hptr.mem_data_len);
    GSCHED_RETAIN;
    *result = cuMemcpyHtoD_v2((CUdeviceptr)dptr, hptr.mem_data_val,
                              hptr.mem_data_len);
    GSCHED_RELEASE;
    return 1;
}

bool_t rpc_culaunchkernel_1_svc(uint64_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, uint64_t hStream, mem_data args, int* result, struct svc_req *rqstp)
{
    GSCHED_RETAIN;
    void **cuda_args;
    uint16_t *arg_offsets;
    size_t param_num;


    GET_CLIENT_DRV(*result);

    if (args.mem_data_val == NULL) {
        LOGE(LOG_ERROR, "param.mem_data_val is NULL");
        *result = CUDA_ERROR_INVALID_VALUE;
        GSCHED_RELEASE;
        return 1;
    }
    if (args.mem_data_len < sizeof(size_t)) {
        LOGE(LOG_ERROR, "param.mem_data_len is too small");
        *result = CUDA_ERROR_INVALID_VALUE;
        GSCHED_RELEASE;
        return 1;
    }
    param_num = *((size_t*)args.mem_data_val);

    if (args.mem_data_len < sizeof(size_t)+sizeof(uint16_t)*param_num) {
        LOGE(LOG_ERROR, "param.mem_data_len is too small");
        *result = CUDA_ERROR_INVALID_VALUE;
        GSCHED_RELEASE;
        return 1;
    }

    arg_offsets = (uint16_t*)(args.mem_data_val+sizeof(size_t));
    cuda_args = malloc(param_num*sizeof(void*));
    for (size_t i = 0; i < param_num; ++i) {
        cuda_args[i] = args.mem_data_val+sizeof(size_t)+param_num*sizeof(uint16_t)+arg_offsets[i];
        *(void**)cuda_args[i] = resource_map_get_addr_default(client->gpu_mem, *(void**)cuda_args[i], *(void**)cuda_args[i]);
        LOGE(LOG_DEBUG, "arg: %p (%d)", *(void**)cuda_args[i], *(int*)cuda_args[i]);
    }

    void *f_ptr = NULL;
    if (resource_mg_get(&client->functions, (void*)f, &f_ptr) != 0) {
        LOGE(LOG_ERROR, "function not found in resource manager");
        *result = CUDA_ERROR_INVALID_HANDLE;
        free(cuda_args);
        GSCHED_RELEASE;
        return 1;
    }

    LOGE(LOG_DEBUG, "cuLaunchKernel(func=%p->%p, gridDim=[%d,%d,%d], blockDim=[%d,%d,%d], args=%p, sharedMem=%d, stream=%p)", f, f_ptr, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, cuda_args, sharedMemBytes, (void*)hStream);
    
    cudaStream_t target_stream = hStream ? resource_map_get_addr(client->custom_streams, (void *)hStream) : client->default_stream;

    LOGE(LOG_DEBUG, "target_stream: %p", target_stream);



    *result = cuLaunchKernel((CUfunction)f_ptr,
                              gridDimX, gridDimY, gridDimZ,
                              blockDimX, blockDimY, blockDimZ,
                              sharedMemBytes,
                              (CUstream)hStream,
                              cuda_args, NULL);
    GSCHED_RELEASE;

    free(cuda_args);
    return 1;

}

bool_t rpc_cudevicegetp2pattribute_1_svc(int attrib, ptr srcDevice, ptr dstDevice, int_result *result, struct svc_req *rqstp)
{
    LOG(LOG_DEBUG, "%s", __FUNCTION__);
    GSCHED_RETAIN;
    result->err = cuDeviceGetP2PAttribute(&result->int_result_u.data, (CUdevice_P2PAttribute)attrib, (CUdevice)srcDevice, (CUdevice)dstDevice);
    GSCHED_RELEASE;
    return 1;
}

/* ################## START OF HIDDEN FUNCTIONS IMPL ######################## */

/*
bool_t rpc_hidden_get_device_ctx_1_svc(int dev, ptr_result *result,
                                     struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);

    result->err = ((int(*)(void**,int))(cd_svc_hidden_get(0,1)))
                                        ((void**)&result->ptr_result_u.ptr, dev);
    return 1;
}
*/
/* This function loads the module (the device code)
 * It could be replaced by cuModuleLoad which loads a module via a path
 * string. However, this seems to only work for cubin files that were
 * compiled using "nvcc -cubin" or extracted from a fatbinary using
 * "cuobjdump -xelf". When this returns 200 or 300 the cubin may be
 * compiled for the wrong compute capabilities (e.g. Pascal needs sm_61 and
 * Turing needs sm_75).
 */
/*bool_t rpc_hidden_get_module_1_svc(uint64_t arg2, uint64_t arg3,
                                   uint64_t arg4, int arg5,
                                   ptr_result *result, struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);
    //TODO: make a parameter. Probably must be globally stored somehow
    //      thread-safety may be an issue
    //char str[] = "/home/eiling/projects/cricket/tests/test_api.1.sm_61.cubin";
    //result->err = cuModuleLoad((CUmodule*)&result->ptr_result_u.ptr, str);

    result->err = ((int(*)(void**,void*,uint64_t,uint64_t,int))
                   (cd_svc_hidden_get(0,5)))
                   ((void**)&result->ptr_result_u.ptr, &arg2, arg3,
                    arg4, arg5);
    return 1;
}

bool_t rpc_hidden_1_1_1_svc(ptr_result *result,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);
    void *l_arg1 = NULL;

    ((int(*)(void**, void**))(cd_svc_hidden_get(1,1)))
                             (&l_arg1, (void**)&result->ptr_result_u.ptr);
    result->err = 0;
    return 1;
}

bool_t rpc_hidden_1_3_1_svc(uint64_t arg1, uint64_t arg2, void* unused,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);

    ((void(*)(uint64_t, uint64_t))(cd_svc_hidden_get(1,3)))
                             (arg1, arg2);
    return 1;
}

bool_t rpc_hidden_1_5_1_svc(ptr_result *result,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);
    void *l_arg1 = NULL;

    ((int(*)(void**, void**))(cd_svc_hidden_get(1,5)))
                             (&l_arg1, (void**)&result->ptr_result_u.ptr);
    result->err = 0;
    return 1;
}

bool_t rpc_hidden_2_1_1_svc(uint64_t arg1, void* unused,
                            struct svc_req *rqstp)
{
    printf("%s\n", __FUNCTION__);

    ((void(*)(uint64_t))(cd_svc_hidden_get(2,1)))
                             (arg1);
    return 1;
}

bool_t rpc_hidden_3_0_1_svc(int arg1, uint64_t arg2, uint64_t arg3,
                            int *result, struct svc_req *rqstp)
{
    printf("%s(%d, %p, %p)\n", __FUNCTION__, arg1, arg2, arg3);
    void *fptr = cd_svc_hidden_get(3,0);
    *result = ((int(*)(int, void*, void*))(fptr))
                      (arg1, &arg2, &arg3);
    return 1;
}

bool_t rpc_hidden_3_2_1_svc(int arg2, uint64_t arg3, mem_result *result,
                            struct svc_req *rqstp)
{
    result->mem_result_u.data.mem_data_val = NULL;
    result->mem_result_u.data.mem_data_len = 0x58;
    printf("%s(%d, %p)\n", __FUNCTION__, arg2, arg3);
    //printf("\tppre %s(nh->%p, %d, nh->%p->%p)\n", __FUNCTION__, result->ptr_result_u.ptr, arg2, (void*)arg3, *(void**)arg3);
    void *fptr = cd_svc_hidden_get(3,2);
    result->err = ((int(*)(void**, int, void*))(fptr))
                             ((void**)&result->mem_result_u.data.mem_data_val, arg2, &arg3);
    void **res = ((void**)result->mem_result_u.data.mem_data_val);
    if (res != 0)
        printf("\t%p, @0x30: %p, @0x40: %p->%p\n", res, res[6], res[8], *(void**)res[8]);
    //printf("\tppost %s(nh->%p, %d, nh->%p->%p)\n", __FUNCTION__, result->ptr_result_u.ptr, arg2, (void*)arg3, *(void**)arg3);
    //printf("\terr: %d, result: %p\n", result->err, result->ptr_result_u.ptr);
    return 1;
}
*/
