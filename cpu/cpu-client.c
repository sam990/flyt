#define _GNU_SOURCE
#include <cuda.h>
#include <driver_types.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>

// For TCP socket
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "cpu-common.h"
#include "cpu-libwrap.h"
#include "cpu-utils.h"
#include "cpu_rpc_prot.h"
#include "list.h"
#include "cpu-elf2.h"
#include "cpu-client-mgr-handler.h"
#ifdef WITH_IB
#include "cpu-ib.h"
#endif // WITH_IB

// static const char* LIBCUDA_PATH = "/lib64/libcuda.so";
const char *LIBCUDA_PATH = "/usr/local/cuda/lib64/libcudart.so";

CLIENT *clnt = NULL;

list kernel_infos = { 0 };

char server[256];
unsigned long vers;

INIT_SOCKTYPE
int connection_is_local = 0;
int shm_enabled = 1;
int initialized = 0;

#ifdef WITH_IB
int ib_device = 0;
#endif // WITH_IB

#ifdef WITH_API_CNT
extern void cpu_runtime_print_api_call_cnt(void);
#endif // WITH_API_CNT

static void rpc_connect(char *server_info) // "server_ip, vers" - `vers` is assigned by the node manager.
{
    int isock;
    struct sockaddr_un sock_un = { 0 };
    struct sockaddr_in sock_in = { 0 };
    struct sockaddr_in local_addr = { 0 };
    struct hostent *hp;
    socklen_t sockaddr_len = sizeof(struct sockaddr_in);
    unsigned long prog = 0;

    splitted_str *splitted = split_string(server_info, ",");
    
    if (splitted == NULL) {
        LOGE(LOG_ERROR, "error splitting server info: %s", server_info);
        exit(1);
    }
    if (splitted->size != 2) {
        LOGE(LOG_ERROR, "error parsing server info: %s", server_info);
        exit(1);
    }

    strcpy(server, splitted->str[0]);

    vers = strtoul(splitted->str[1], NULL, 10); // rpc vers is passed back to client for portmapper 

    free_splitted_str(splitted);
    splitted = NULL;
    
    LOG(LOG_INFO, "connection to host \"%s\"", server);

#ifdef WITH_IB

    if (getenv("IB_DEVICE_ID")) {
        ib_device = atoi(getenv("IB_DEVICE_ID"));
    }
    LOG(LOG_INFO, "Using IB device: %d.", ib_device);

#endif // WITH_IB

    prog = 99;

    char *cmd = NULL;
    if (cpu_utils_command(&cmd) != 0) {
        LOGE(LOG_ERROR, "error getting command");
    } else {
        LOG(LOG_DEBUG, "the command is \"%s\"", cmd);
    }
    free(cmd);

    LOGE(LOG_DEBUG, "using prog=%d, vers=%d", prog, vers);

    switch (socktype) {
    case UNIX:
        LOG(LOG_INFO, "connecting via UNIX...");
        isock = RPC_ANYSOCK;
        sock_un.sun_family = AF_UNIX;
        strcpy(sock_un.sun_path, CD_SOCKET_PATH);
        clnt = clntunix_create(&sock_un, prog, vers, &isock, 0, 0);
        connection_is_local = 1;
        break;
    case TCP:
        LOG(LOG_INFO, "connecting via TCP...");
        isock = RPC_ANYSOCK;
        sock_in.sin_family = AF_INET;
        sock_in.sin_port = 0; // ensures libtirpc routes to portmapper.
        if ((hp = gethostbyname(server)) == 0) {
            LOGE(LOG_ERROR, "error resolving hostname: %s", server);
            exit(1);
        }
        // contains IP of host, 
        // no resolve reqd by gethostbyname
        // as `server` is an IPV4 address.
        sock_in.sin_addr = *(struct in_addr *)hp->h_addr;
        // inet_aton("137.226.133.199", &sock_in.sin_addr);

        // create a TCP client handle that connects to 
        // a single cricket-rpc-server socket.
        // isock: socket that is bound to the server IP.
        // sock_in: sockaddr_in struct that contains server IP.
        clnt = clnttcp_create(&sock_in, prog, vers, &isock, 0, 0);
        getsockname(isock, &local_addr, &sockaddr_len);
        connection_is_local =
            (local_addr.sin_addr.s_addr == sock_in.sin_addr.s_addr);
        break;
    case UDP:
        /* From RPCEGEN documentation:
         * Warning: since UDP-based RPC messages can only hold up to 8 Kbytes
         * of encoded data, this transport cannot be used for procedures that
         * take large arguments or return huge results.
         * -> Sounds like UDP does not make sense for CUDA, because we need to
         *    be able to copy large memory chunks
         **/
        printf("UDP is not supported...\n");
        break;
    }

    if (clnt == NULL) {
        clnt_pcreateerror("[rpc] Error");
        exit(1);
    }
}

void change_server(char *server_info)
{
    enum clnt_stat retval_1;
    int result_1;
    LOG(LOG_INFO, "changing server to %s", server_info);
    rpc_connect(server_info);
}

void resume_connection(void)
{
    enum clnt_stat retval_1;
    int result_1;
    retval_1 = rpc_ckp_restore_1(getpid(), &result_1, clnt);
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "error calling rpc_restore");
    }
}

// static void repair_connection(int signo)
// {
//     enum clnt_stat retval_1;
//     int result_1;
//     /*LOGE(LOG_INFO, "Trying connection...");
//     char *printmessage_1_arg1 = "connection test";
//     FUNC_BEGIN 
//     retval_1 = rpc_printmessage_1(printmessage_1_arg1, &result_1, clnt);
//     FUNC_END
//     printf("return:%d\n", result_1);
//     if (retval_1 == RPC_SUCCESS) {
//         LOG(LOG_INFO, "connection still okay.");
//         return;
//     }*/
//     LOG(LOG_INFO, "connection dead. Reconnecting...");
//     rpc_connect();
//     LOG(LOG_INFO, "reconnected");
//     FUNC_BEGIN 
//     retval_1 = cuda_device_synchronize_1(&result_1, clnt);
//     FUNC_END
//     if (retval_1 != RPC_SUCCESS) {
//         LOGE(LOG_ERROR, "error calling cudaDeviceSynchronize");
//     }
// }

// Called as soon as the library is loaded.
void __attribute__((constructor)) init_rpc(void)
{
    enum clnt_stat retval_1;
    int result_1;
    int_result result_2;
    char *printmessage_1_arg1 = "hello";

    LOG(LOG_DBG(1), "log level is %d", LOG_LEVEL);
    init_log(LOG_LEVEL, __FILE__);

    pthread_rwlock_init(&access_sem, NULL);

    char* server_info = init_client_mgr();
    if (server_info == NULL) {
        LOGE(LOG_ERROR, "error initializing client manager");
        exit(1);
    }

    rpc_connect(server_info);
    free(server_info);

    initialized = 1;

    /// Now we can communicate ivshmem params to the server.
    /// In rpc_init_1, need to additionally send to server:
    // - path to ivshmem backend. ("/dev/shm/<qemu-ivshmem-name>", string)
    // - Size of ivshmem backend allocated to this process. (constant per process, int)
    // - Whether client VM and flyt-rpc-server are on the same machine. (0/1, int)
    // - Starting byte offset into the backend file for this process. ()
    /// The client library will maintain these details on the VM as well.
    /// Each process will have its own instance of the tracking vars used
    /// in the lib.



    FUNC_BEGIN 
    retval_1 = rpc_init_1(getpid(), &result_1,  clnt);
    FUNC_END
    
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror(clnt, "call failed");
    }

    if (result_1 != 0) {
        LOGE(LOG_ERROR, "cricket initialisation failed");
        exit(1);
    }

    if (list_init(&kernel_infos, sizeof(kernel_info_t)) != 0) {
        LOGE(LOG_ERROR, "list init failed.");
    }

    if (elf2_init() != 0) {
        LOGE(LOG_ERROR, "libelf init failed");
    }

    // if (cpu_utils_parameter_info(&kernel_infos, "/proc/self/exe") != 0) {
    //     LOG(LOG_ERROR, "error while getting parameter size. Check whether "
    //                    "cuobjdump binary is in PATH! Trying anyway (will only "
    //                    "work if there is no kernel in this binary)");
    // }
#ifdef WITH_IB
    if (ib_init(ib_device, server) != 0) {
        LOG(LOG_ERROR, "initilization of infiniband verbs failed.");
    }
#endif // WITH_IB
}
void __attribute__((destructor)) deinit_rpc(void)
{
    enum clnt_stat retval_1;
    int result;
    if (initialized) {
        FUNC_BEGIN 
        retval_1 = rpc_deinit_1(&result, clnt);
        FUNC_END
        if (retval_1 != RPC_SUCCESS) {
            LOGE(LOG_ERROR, "call failed.");
        }
        kernel_infos_free(kernel_infos.elements, kernel_infos.length);
        list_free(&kernel_infos);
#ifdef WITH_API_CNT
        cpu_runtime_print_api_call_cnt();
#endif // WITH_API_CNT
    }

    if (clnt != NULL) {
       clnt_destroy(clnt);
    }
}


static void *(*dlopen_orig)(const char *, int) = NULL;
static int (*dlclose_orig)(void *) = NULL;
static void *dl_handle = NULL;

void *dlopen(const char *filename, int flag)
{
    void *ret = NULL;
    struct link_map *map;
    int has_kernel = 0;
    LOG(LOG_DBG(1), "intercepted dlopen(%s, %d)", filename, flag);

    if (filename == NULL) {
        return dlopen_orig(filename, flag);
    }

    if (dlopen_orig == NULL) {
        if ((dlopen_orig = dlsym(RTLD_NEXT, "dlopen")) == NULL) {
            LOGE(LOG_ERROR, "[dlopen] dlsym failed");
        }
    }

    static const char *replace_libs[] = {
        "libcuda.so.1",
        "libcuda.so",
        "libnvidia-ml.so.1",
        "libcudnn_cnn_infer.so.8"
    };
    static const size_t replace_libs_sz = sizeof(replace_libs) / sizeof(char *);
    if (filename != NULL) {
        for (size_t i=0; i != replace_libs_sz; ++i) {
            if (strcmp(filename, replace_libs[i]) == 0) {
                LOG(LOG_DEBUG, "replacing dlopen call to %s with cricket-client.so", filename);
                dl_handle = dlopen_orig("cricket-client.so", flag);
                if (clnt == NULL) {
                    LOGE(LOG_WARNING, "rpc seems to be uninitialized while loading %s", filename);
                }
                return dl_handle;
            }
        }
    }
    /* filename is NULL or not in replace_libs list */
    if ((ret = dlopen_orig(filename, flag)) == NULL) {
        LOGE(LOG_ERROR, "dlopen %s failed: ", filename, dlerror());
    } else if (has_kernel) {
        dlinfo(ret, RTLD_DI_LINKMAP, &map);
        LOGE(LOG_DEBUG, "dlopen to  %p", map->l_addr);
    }
    return ret;
}

int dlclose(void *handle)
{
    if (handle == NULL) {
        LOGE(LOG_ERROR, "[dlclose] handle NULL");
        return -1;
    } else if (dlclose_orig == NULL) {
        if ((dlclose_orig = dlsym(RTLD_NEXT, "dlclose")) == NULL) {
            LOGE(LOG_ERROR, "[dlclose] dlsym failed");
        }
    }

    // Ignore dlclose call that would close this library
    if (dl_handle == handle) {
        LOGE(LOG_DEBUG, "[dlclose] ignore close");
        return 0;
    } else {
        return dlclose_orig(handle);
    }
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size, int constant,
                       int global)
{
    enum clnt_stat retval_1;
    int result;
    LOGE(LOG_DEBUG, "__cudaRegisterVar(fatCubinHandle=%p, hostVar=%p, deviceAddress=%p, "
           "deviceName=%s, ext=%d, size=%zu, constant=%d, global=%d)\n",
           fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    FUNC_BEGIN 
    retval_1 = rpc_register_var_1((ptr)fatCubinHandle, (ptr)hostVar, (ptr)deviceAddress, (char*)deviceName, ext, size, constant, global,
                                       &result, clnt);
    FUNC_END
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed.");
    }
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize)
{
    ptr_result result;
    enum clnt_stat retval_1;

    LOGE(LOG_DEBUG, "__cudaRegisterFunction(fatCubinHandle=%p, hostFun=%p, devFunc=%s, "
           "deviceName=%s, thread_limit=%d, tid=[%p], bid=[%p], bDim=[%p], "
           "gDim=[%p], wSize=%p)\n",
           fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
           bid, bDim, gDim, wSize);

    kernel_info_t *info = utils_search_info(&kernel_infos, (char *)deviceName);
    if (info == NULL) {
        LOGE(LOG_ERROR, "request to register unknown function: \"%s\"",
             deviceName);
        return;
    } else {
        LOGE(LOG_DEBUG, "request to register known function: \"%s\"",
             deviceName);
        FUNC_BEGIN 
        retval_1 = rpc_register_function_1((ptr)fatCubinHandle, (ptr)hostFun,
                                           deviceFun, (char*)deviceName, thread_limit,
                                           &result, clnt);
        FUNC_END
        if (retval_1 != RPC_SUCCESS) {
            LOGE(LOG_ERROR, "call failed.");
            exit(1);
        }
        if (result.err != 0) {
            LOGE(LOG_ERROR, "error registering function: %d", result.err);
            exit(1);
        }
        info->host_fun = (void *)hostFun;
    }
}


void **__cudaRegisterFatBinary(void *fatCubin)
{
    void **result;
    int rpc_result;
    enum clnt_stat retval_1;
    size_t fatbin_size;
    LOGE(LOG_DEBUG, "__cudaRegisterFatBinary(fatCubin=%p)", fatCubin);

    mem_data rpc_fat = { .mem_data_len = 0, .mem_data_val = NULL };

    if (elf2_get_fatbin_info((struct fat_header *)fatCubin,
                                &kernel_infos,
                                (uint8_t **)&rpc_fat.mem_data_val,
                                &fatbin_size) != 0) {
        LOGE(LOG_ERROR, "error getting fatbin info");
        return NULL;
    }
    rpc_fat.mem_data_len = fatbin_size;

    // CUDA registers an atexit handler for fatbin cleanup that accesses
    // the fatbin data structure. Let's allocate some zeroes to avoid segfaults.
    result = (void**)calloc(1, 0x58);

    FUNC_BEGIN 
    retval_1 = rpc_elf_load_1(rpc_fat, (ptr)result, &rpc_result, clnt);
    FUNC_END
    if (retval_1 != RPC_SUCCESS) {
        LOGE(LOG_ERROR, "call failed.");
    }
    if (rpc_result != 0) {
        LOGE(LOG_ERROR, "error registering fatbin: %d", rpc_result);
        return NULL;
    }
    LOG(LOG_DEBUG, "fatbin loaded to %p", result);
    // we return a bunch of zeroes to avoid segfaults. The memory is
    // mapped by the modules resource 
    return result;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{  
    int result;
    enum clnt_stat retval_1;

    LOGE(LOG_DEBUG, "__cudaUnregisterFatBinary(fatCubinHandle=%p)",
         fatCubinHandle);

    if (fatCubinHandle == NULL) {
        LOGE(LOG_WARNING, "fatCubinHandle is NULL - so we have nothing to unload. (This is okay if this binary does not contain a kernel.)");
        return;
    }

    // retval_1 = rpc_elf_unload_1((ptr)fatCubinHandle, &result, clnt);
    // if (retval_1 != RPC_SUCCESS || result != 0) {
    //     LOGE(LOG_ERROR, "call failed.");
    // }
}

// void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
// {
//     int result;
//     enum clnt_stat retval_1;

//     //printf("__cudaRegisterFatBinaryEnd(fatCubinHandle=%p)\n",
//     fatCubinHandle);

//     retval_1 =
//     RPC_SUCCESS;//cuda_register_fat_binary_end_1((uint64_t)fatCubinHandle,
//     &result, clnt); if (retval_1 != RPC_SUCCESS) {
//         clnt_perror (clnt, "call failed");
//     }
// }
