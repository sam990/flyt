#include "cpu-server-client-mgr.h"
#include "resource-map.h"
#include "cpu_rpc_prot.h"
#include "cpu-server-runtime.h"


typedef struct __mem_ckp_header_t {
    void *haddr;
    mem_alloc_args_t args;
} mem_ckp_header_t;

int flyt_create_checkpoint(char *basepath);

int flyt_restore_checkpoint(char *basepath);
