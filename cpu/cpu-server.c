#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h> //unlink()
#include <signal.h> //sigaction
#include <sys/types.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <link.h>

#include "cpu-server.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"
#include "log.h"
#include "cpu-server-runtime.h"
#include "cpu-server-driver.h"
#include "rpc/xdr.h"
#include "cr.h"
#include "cpu-elf2.h"
#ifdef WITH_IB
#include "cpu-ib.h"
#endif //WITH_IB
#define WITH_RECORDER
#include "api-recorder.h"
#include "gsched.h"
#include "cpu-server-nvml.h"
#include "cpu-server-cudnn.h"
#include "cpu-server-mgr-listener.h"
#include "cpu-server-resource-controller.h"
#include "cpu-server-client-mgr.h"
#include "cpu-server-ivshmem.h"
#include "cpu-server-dev-mem.h"
#include "cpu-server-runtime-rpc-shm.h"

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif


INIT_SOCKTYPE

int connection_is_local = 0;
int shm_enabled = 1;

int got_request = 0;

#ifdef WITH_IB
    int ib_device = 0;
#endif //WITH_IB

extern gsched_fixed_t sched_fixed;

unsigned long prog=0, vers=0;

extern void rpc_cd_prog_1(struct svc_req *rqstp, register SVCXPRT *transp);

void int_handler(int signal) {
    if (socktype == UNIX) {
        unlink(CD_SOCKET_PATH);
    }

    LOG(LOG_INFO, "have a nice day!\n");
    svc_exit();
}

bool_t rpc_printmessage_1_svc(char *argp, int *result, struct svc_req *rqstp)
{
    LOG(LOG_INFO, "string: \"%s\"\n", argp);
    *result = 42;
    return 1;
}

int poll_pause = 0;
int do_remove_client = 0;
int shm_exists = 0; // if one client has shm, all will have both tcp and shm

pthread_cond_t rm_client_cond_var = PTHREAD_COND_INITIALIZER;
pthread_mutex_t rm_client_mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t poll_pause_cond_var = PTHREAD_COND_INITIALIZER;
pthread_mutex_t poll_pause_mutex = PTHREAD_MUTEX_INITIALIZER;

void *do_rpc_shm_poll_svc() {
    cricket_client * (*iter_fn) (cricket_client_iter *) = get_next_client;
    while(1) {
        // pthread_mutex_lock(&poll_pause_mutex);
        // // while (poll_pause) {
        // //     printf("pausing poll...\n");
        // //     //pthread_cond_signal(&rm_client_cond_var);

        // //     pthread_cond_wait(&poll_pause_cond_var, &poll_pause_mutex);
        // // }
        // // // poll_pause = 0;
        // // pthread_mutex_unlock(&poll_pause_mutex);
        cricket_client_iter iter = get_client_iter();
        cricket_client *client;
        
        int _cnt = 0;
        while ((client = iter_fn(&iter)) != NULL) {
            //printf("polling on server, client %d\n", _cnt);
            if (client->ivshmem_ctx) {
                // check for notif at poll_s
                uint8_t *notif = (uint8_t *)client->ivshmem_ctx->shm_mmap + 1;
                //printf("Notif value in poll: %d\n", *notif);

                if (*notif == 1) {
                    //printf("msg from client!\n");
                    pthread_mutex_lock(&(client->ivshmem_ctx->poll_mutex_svc));
                    got_request = 1;
                    pthread_cond_signal(&(client->ivshmem_ctx->poll_cond_var_svc));
                    pthread_mutex_unlock(&(client->ivshmem_ctx->poll_mutex_svc));
                }
                _cnt++;
            }
        }
        // printf("polling on server\n");
        usleep(1);
    }
}

void begin_poll_svc() {
    pthread_t poll_tid;
    pthread_create(&poll_tid, NULL, do_rpc_shm_poll_svc, NULL);
    pthread_detach(poll_tid);
}

// one per client.
void *rpc_shm_dispatcher(void *arg) {
    cricket_client *client = (cricket_client *)arg;
    printf("Server waiting for notif, client %d\n", client->pid);
    while(1) {
        pthread_mutex_lock(&(client->ivshmem_ctx->poll_mutex_svc));
        while (!got_request) {
            pthread_cond_wait(&(client->ivshmem_ctx->poll_cond_var_svc), &(client->ivshmem_ctx->poll_mutex_svc));
        }
        got_request = 0;
        pthread_mutex_unlock(&(client->ivshmem_ctx->poll_mutex_svc));

        rpc_shm_header_t *rpc_hdr_svc = (rpc_shm_header_t *)(client->ivshmem_ctx->shm_mmap);

        // clear poll_s
        uint8_t *notif = (uint8_t *)client->ivshmem_ctx->shm_mmap + 1;
        *notif = 0;

        // read the cmd
        //printf("cmd: %d\nFor client %d\n", rpc_hdr_svc->rpc_cmd, client->pid);
        switch (rpc_hdr_svc->rpc_cmd) {
            case CUDA_GET_DEVICE_COUNT: {
                int_result result;
                rpc_shm_svc_cuda_get_device_count_1(rpc_hdr_svc, &result, client);
                // notify
                break;
            }
            
            case CUDA_GET_DEVICE_PROPERTIES: {
                // result struct will be memcpied to some location, 
                // and offset will be written to the result field
                // before notify.
                int_result result;
                rpc_shm_svc_cuda_get_device_properties_1(rpc_hdr_svc, &result, client); 
                break;
            }
        }
    }
}

bool_t rpc_deinit_1_svc(int *result, struct svc_req *rqstp)
{
    LOG(LOG_INFO, "RPC deinit requested.");
    //svc_exit();

    // stop poll by signal
    // pthread_mutex_lock(&poll_pause_mutex);
    // poll_pause = 1;
    // // pthread_cond_signal(&poll_pause_cond_var);
    // pthread_mutex_unlock(&poll_pause_mutex);

    // wait till signal on condvar1
    // pthread_mutex_lock(&rm_client_mutex);
    // while(!do_remove_client) {
    //     pthread_cond_wait(&rm_client_cond_var, &rm_client_mutex);
    // }
    // pthread_mutex_unlock(&rm_client_mutex);

    // now you can remove client.
    // remove the tcp handle
    remove_client(rqstp->rq_xprt->xp_fd);

    // remove its shm handle
    // if (shm_exists) {
    //     remove_client(-rqstp->rq_xprt->xp_fd);
    // }

    // finally, signal condvar2 for threads to resume
    // pthread_mutex_lock(&poll_pause_mutex);
    // poll_pause = 0;
    // pthread_mutex_unlock(&poll_pause_mutex);
    // pthread_cond_signal(&poll_pause_cond_var);
    printf("signal to poll done\n");

    return 1;
}

int proc_cnt = 0;
bool_t rpc_init_1_svc(int pid, ivshmem_setup_desc iv_stat, int *result, struct svc_req *rqstp) {
    LOG(LOG_INFO, "RPC init requested %d", rqstp->rq_xprt->xp_fd);
    
    // create and initialize 
    cricket_client *client_tcp = NULL;
    cricket_client *client_shm = NULL;
    ivshmem_svc_ctx *ivshmem_ctx = NULL;
    if (iv_stat.iv_enable == 1) {
        shm_exists = 1;
        char *f_be = iv_stat.f_be;
        LOGE(LOG_DEBUG, "ivshmem backend file: %s\n", f_be);
        LOGE(LOG_DEBUG, "ivshmem enable?: %d\n", iv_stat.iv_enable);
        // shm OK
        ivshmem_ctx = init_ivshmem_svc(iv_stat); // mark the region with the pid of the process in shm.

        // create client with negative xp_fd
        int rpc_shm_xp_fd = -(rqstp->rq_xprt->xp_fd);
        client_shm = add_new_client(pid, rpc_shm_xp_fd, ivshmem_ctx); 

        // EVERY SHM CLIENT WILL HAVE A TCP HANDLE ALSO.
        // IT WILL BE STORED WITH PID = -(ACTUAL PID)
        // GET_CLIENT ONLY USES XP_FD, SO PID IS IRRELEVANT FOR TCP DISPATCH
        client_tcp = add_new_client(-pid, rqstp->rq_xprt->xp_fd, NULL); 

        if (!(client_shm && client_tcp)) {
            LOGE(LOG_ERROR, "Failed to initialize client manager for pid %d", pid);
            *result = 1;
            return 1;
        }

        // create a new thread (for the new client) that is waiting on a condition.
        // variable signalled by the polling thread. Pass the client struct to the
        // thread function (rpc_shm_worker())
        // these handler threads must call the cuda_rpc funcs in server_runtime.
        pthread_t worker_thread;
        int t_ret = pthread_create(&worker_thread, NULL, rpc_shm_dispatcher, (void *)client_shm);
        if (t_ret != 0) {
            printf("Failed to create thread: %d\n", t_ret);
            return -1;
        }
        pthread_detach(worker_thread);
        __sync_synchronize(); // ensure worker thread created before poll starts.

        // if first process, create a new polling thread. (the function is in an infinite loop)
        // - if write detected, check the pid of the process.
        // - wake up (signal) only the thread that is handling this pid. ?? How?
        // - Threads are created dynamically, cannot have a fixed number of condition variables.
        if (!proc_cnt) {
            proc_cnt++;
            begin_poll_svc(); // start the thread. Needs to be able to poll every ivshmem_ctx poll region.
        }
    } else {
        client_tcp = add_new_client(pid, rqstp->rq_xprt->xp_fd, ivshmem_ctx); 
        if (!client_tcp) {
            LOGE(LOG_ERROR, "Failed to initialize client manager for pid %d", pid);
            *result = 1;
            return 1;
        }
        // if tcp
        do_remove_client = 1;
    }

    *result = 0;

    return 1;
}

bool_t rpc_ckp_restore_1_svc(int pid, int *result, struct svc_req *rqstp) {
    LOG(LOG_INFO, "RPC ckp_restore requested %d", rqstp->rq_xprt->xp_fd);
    // rqstp->rq_xprt->xp_fd 
    int ret = move_restored_client(pid, rqstp->rq_xprt->xp_fd);
    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to initialize client manager for pid %d", pid);
        *result = 1;
        return 1;
    }
    *result = 0;
    return 1;
}

int cricket_server_checkpoint(int dump_memory)
{
    int ret;
    struct stat path_stat = { 0 };
    const char *ckp_path = "ckp";
    LOG(LOG_INFO, "rpc_checkpoint requested.");

    if (!ckp_path) {
        LOGE(LOG_ERROR, "ckp_path is NULL");
        goto error;
    }
    if (stat(ckp_path, &path_stat) != 0) {
        LOG(LOG_DEBUG, "directory \"%s\" does not exist. Let's create it.", ckp_path);
        if (mkdir(ckp_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            LOGE(LOG_ERROR, "failed to create directory \"%s\"", ckp_path);
            goto error;
        }
    } else if (!S_ISDIR(path_stat.st_mode)) {
        LOG(LOG_ERROR, "file \"%s\" is not a directory", ckp_path);
        goto error;
    }
    
    // if ((ret = server_runtime_checkpoint(ckp_path, dump_memory, prog, vers)) != 0) {
    //     LOGE(LOG_ERROR, "server_runtime_checkpoint returned %d", ret);
    //     goto error;
    // }

    LOG(LOG_INFO, "checkpoint successfully created.");
    return 0;
 error:
    LOG(LOG_INFO, "checkpoint creation failed.");
    return 1;
}

static void signal_checkpoint(int signo)
{
    if (cricket_server_checkpoint(1) != 0) {
        LOGE(LOG_ERROR, "failed to create checkpoint");
    }
}

bool_t rpc_checkpoint_1_svc(int *result, struct svc_req *rqstp)
{
    int ret;
    if ((ret = cricket_server_checkpoint(1)) != 0) {
        LOGE(LOG_ERROR, "failed to create checkpoint");
    }
    return ret == 0;
}

/* Call CUDA initialization function (usually called by __libc_init_main())
* Address of "_ZL24__sti____cudaRegisterAllv" in static symbol table is e.g. 0x4016c8
*/
void cricket_so_register(void* dlhandle, char *path)
{
    // struct link_map *map;
    // dlinfo(dlhandle, RTLD_DI_LINKMAP, &map);

    // // add load location of library to offset in symbol table
    // void (*cudaRegisterAllv)(void) = 
    //     (void(*)(void)) elf_symbol_address(path, "_ZL24__sti____cudaRegisterAllv");
    
    // LOG(LOG_INFO, "found CUDA initialization function at %p + %p = %p", 
    //     map->l_addr, cudaRegisterAllv, map->l_addr + cudaRegisterAllv);

    // cudaRegisterAllv += map->l_addr;
    
    // if (cudaRegisterAllv == NULL) {
    //     LOGE(LOG_WARNING, "could not find cudaRegisterAllv initialization function in cubin. Kernels cannot be launched without it!");
    // } else {
    //     cudaRegisterAllv();
    // }
}

bool_t rpc_dlopen_1_svc(char *path, int *result, struct svc_req *rqstp)
{
    void *dlhandle;

    if (path == NULL) {
        LOGE(LOG_ERROR, "path is NULL");
        *result = 1;
        return 1;
    }
    if ((dlhandle = dlopen(path, RTLD_LAZY)) == NULL) {
        LOGE(LOG_ERROR, "error opening \"%s\": %s. Make sure libraries are present.", path, dlerror());
        *result = 1;
        return 1;
    } else {
        LOG(LOG_INFO, "dlopened \"%s\"", path);

       //cricket_so_register(dlhandle, path);

    }
    *result = 0;
    return 1;
}

void cricket_main(size_t prog_num, size_t vers_num, uint32_t gpu_id, uint32_t num_sm_cores, size_t memory, uint32_t thread_mode)
{
    int ret = 1;
    register SVCXPRT *transp;

    int protocol = 0;
    int restore = 0;
    struct sigaction act;
    char *command = NULL;
    act.sa_handler = int_handler;
    printf("welcome to cricket!\n");
    //init_log(LOG_LEVEL, __FILE__);
    LOG(LOG_DBG(1), "log level is %d", LOG_LEVEL);
    sigaction(SIGINT, &act, NULL);

    #ifdef WITH_IB
    char client[256];
    char envvar[] = "CLIENT_ADDRESS";

    if(!getenv(envvar)) {
        LOG(LOG_ERROR, "Environment variable %s does not exist. For memory transports using InfiniBand it must contain the address of the client.", envvar);
        exit(1);
    }
    if(strncpy(client, getenv(envvar), 256) == NULL) {
        LOGE(LOG_ERROR, "strncpy failed.");
        exit(1);
    }
    LOG(LOG_INFO, "connection to client \"%s\"", client);

    if(getenv("IB_DEVICE_ID")) {
        ib_device = atoi(getenv("IB_DEVICE_ID"));
    }
    LOG(LOG_INFO, "Using IB device: %d.", ib_device);

    #endif //WITH_IB


    if (getenv("CRICKET_DISABLE_RPC")) {
        LOG(LOG_INFO, "RPC server was disable by setting CRICKET_DISABLE_RPC");
        return;
    }
    if (getenv("CRICKET_RESTORE")) {
        LOG(LOG_INFO, "restoring previous state was enabled by setting CRICKET_RESTORE");
            restore = 1;
    }

    if (restore == 1) {
        if (cr_restore_rpc_id("ckp", &prog, &vers) != 0) {
            LOGE(LOG_ERROR, "error while restoring rpc id");
        }
    } else {
        prog = prog_num;
        vers = vers_num;
    }

    LOGE(LOG_DEBUG, "using prog=%d, vers=%d", prog, vers);


    switch (socktype) {
    case UNIX:
        LOG(LOG_INFO, "using UNIX...");
        transp = svcunix_create(RPC_ANYSOCK, 0, 0, CD_SOCKET_PATH);
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        }
        connection_is_local = 1;
        break;
    case TCP:
        LOG(LOG_INFO, "using TCP...");
        // a random port number is assigned to this socket,
        // now strored in a handler of type SVCXPRT *
        // ----------------------------------------
        // - Consider SVCXPRT == __rpc_svcxprt as a 
        // generic transport handler class with
        transp = svctcp_create(RPC_ANYSOCK, 0, 0);
        // Now we have a listen fd being tracked by select
        // for connection request events.
        if (transp == NULL) {
            LOGE(LOG_ERROR, "cannot create service.");
            exit(1);
        }
        // remove any previous mappings of this program
        // on the rpcbind service in preperation for this 
        // new svc_registration.
        pmap_unset(prog, vers);
        LOG(LOG_INFO, "listening on port %d", transp->xp_port);
        protocol = IPPROTO_TCP;
        break;
    case UDP:
        /* From RPCGEN documentation:
         * Warning: since UDP-based RPC messages can only hold up to 8 Kbytes
         * of encoded data, this transport cannot be used for procedures that
         * take large arguments or return huge results.
         * -> Sounds like UDP does not make sense for CUDA, because we need to
         *    be able to copy large memory chunks
         **/
        LOG(LOG_INFO, "UDP is not supported...");
        break;
    }

    // "service" - implemented by a program.
    // A "program" - implements several remote procedures.
    // Thus an RPC service is implemented by a server that implements
    // a set of remote procedures.
    // ---------------------------
    // - struct svc_callout: Implements the list of registered callouts
    if (!svc_register(transp, prog, vers, rpc_cd_prog_1, protocol)) {
        LOGE(LOG_ERROR, "unable to register (RPC_PROG_PROG, RPC_PROG_VERS).");
        exit(1);
    }

    // Set the maximum size of incoming requests
    if (thread_mode == 1) {
	    int mode = RPC_SVC_MT_AUTO; //51
	    printf("thread mode = %d\n", RPC_SVC_MTMODE_SET);
        if (rpc_control(RPC_SVC_MTMODE_SET, (void *)&mode) != TRUE) {
            LOGE(LOG_ERROR, "unable to set multi threaded mode .\n");
            exit(1);
        }
    }

    /* Call CUDA initialization function (usually called by __libc_init_main())
     * Address of "_ZL24__sti____cudaRegisterAllv" in static symbol table is e.g. 0x4016c8
     */
    // void (*cudaRegisterAllv)(void) =
    //     (void(*)(void)) elf_symbol_address(NULL, "_ZL24__sti____cudaRegisterAllv");
    // LOG(LOG_INFO, "found CUDA initialization function at %p", cudaRegisterAllv);
    // if (cudaRegisterAllv == NULL) {
    //     LOGE(LOG_WARNING, "could not find cudaRegisterAllv initialization function in cubin. Kernels cannot be launched without it!");
    // } else {
    //     cudaRegisterAllv();
    // }

    sched = &sched_fixed; // a struct of function pointers defined in gsched_fixed.c
    set_active_device(gpu_id);

    if (sched->init() != 0) {
        LOGE(LOG_ERROR, "initializing scheduler failed.");
        goto cleanup4;
    }
    // store call history on server in list api_records.
    if (list_init(&api_records, sizeof(api_record_t)) != 0) {
        LOGE(LOG_ERROR, "initializing api recorder failed.");
        goto cleanup4;
    }
    
    // init resource manager lists for runtime api calls
    if (server_runtime_init(restore, gpu_id) != 0) {
        LOGE(LOG_ERROR, "initializing server_runtime failed.");
        goto cleanup3;
    }
    // do nothing for now
    if (server_driver_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_runtime failed.");
        goto cleanup2;        
    }
    
    if (server_nvml_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_nvml failed.");
        goto cleanup1;
    }

    if (server_cudnn_init(restore) != 0) {
        LOGE(LOG_ERROR, "initializing server_nvml failed.");
        goto cleanup0;
    }

#ifdef WITH_IB

    if (ib_init(ib_device, client) != 0) {
        LOG(LOG_ERROR, "initilization of infiniband verbs failed.");
        goto cleanup1;
    }
    
#endif // WITH_IB


    // if (signal(SIGUSR1, signal_checkpoint) == SIG_ERR) {
    //     LOGE(LOG_ERROR, "An error occurred while setting a signal handler.");
    //     goto cleanup00;
    // }


    init_listener(vers);
    if (init_resource_controller(num_sm_cores, memory) != 0) {
        LOGE(LOG_ERROR, "initializing resource controller failed.");
        goto cleanup00;
    }

    if (init_cpu_server_client_mgr() != 0) {
        LOGE(LOG_ERROR, "initializing client manager failed.");
        goto cleanup00;
    }

    if (init_server_dev_mem(gpu_id) != 0) {
        LOGE(LOG_ERROR, "initializing server_dev_mem failed.");
        goto cleanup00;
    }

    LOG(LOG_INFO, "waiting for RPC requests...");

    // make sure that our output is flushed even for non line-buffered shells
    fflush(stdout);
    
    // to signal back to client via node manager that the server is ready 
    // to accept connection requests.
    send_initialised_msg(); 

    if(thread_mode == 1) {
	    int mt_mode = RPC_SVC_MT_AUTO;
	    rpc_control(RPC_SVC_MTMODE_SET, &mt_mode);
    }
    else {
	    int mt_mode = RPC_SVC_MT_NONE;
	    rpc_control(RPC_SVC_MTMODE_SET, &mt_mode);
    }

    svc_run(); // libtirpc. rpc messages can now successfully arrive and call the dispatch.

    LOG(LOG_DEBUG, "svc_run returned. Cleaning up.");
    ret = 0;
    //api_records_print();
 cleanup00:
    server_cudnn_deinit();
 cleanup0:
    server_driver_deinit();
 cleanup1:
    server_nvml_deinit();
 cleanup2:
    server_runtime_deinit();
 cleanup3:
    api_records_free();
 cleanup4:
    pmap_unset(prog, vers);
    svc_destroy(transp);
    unlink(CD_SOCKET_PATH);
    LOG(LOG_DEBUG, "have a nice day!");
    exit(ret);
}

int rpc_cd_prog_1_freeresult (SVCXPRT * a, xdrproc_t b , caddr_t c)
{
    if (b == (xdrproc_t) xdr_str_result) {
        str_result *res = (str_result*)c;
        if (res->err == 0) {
            free( res->str_result_u.str);
        }
    }
    else if (b == (xdrproc_t) xdr_mem_result) {
        mem_result *res = (mem_result*)c;
        if (res->err == 0) {
            free( (void*)res->mem_result_u.data.mem_data_val);
        }
    }
    return 1;
}

