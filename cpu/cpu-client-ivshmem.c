/**
 * FILE: cpu-client-ivshmem.c
 * --------------------------
 * Implements ivshmem related operations
 *
 */

#include "cpu-client-ivshmem.h"
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

ivshmem_clnt_ctx *ivshmem_ctx = NULL; 

// enter only if shm_enabled.
void init_ivshmem_clnt(int clnt_pid, char *shm_be_path) {
    assert(ivshmem_ctx == NULL);

    ivshmem_ctx = malloc(sizeof(ivshmem_clnt_ctx));
    assert(ivshmem_ctx != NULL);
    ivshmem_ctx->shm_enabled = SHM_OK;
    ivshmem_ctx->pid = clnt_pid;


    // ivshmem_setup_desc _args = clnt_mgr_get_shm(clnt_pid, shm_be_path);

    // get ivshmem_args from client manager via UDS
    // currently hardcoded default.
    // These can be stored in the MongoDB database initially
    // and then transferred to the client manager.
    // be_off can be updated by init_ivshmem_clnt
    // --
    // hardcoding offsets only works for single-proc, single-VM case.
    #define PROC_SHM_SIZE 2000000
    #define PROC_WRITE_START_OFFSET_CLNT (PROC_SHM_SIZE / 2)
    #define PROC_READ_START_OFFSET_CLNT 0
    ivshmem_setup_desc _args = {
        .iv_enable = 1, // shm_enabled, IN MONGO
        .f_be = "/dev/shm/ivshmem-0-ub11.dat", // IN MONGO
        .proc_be_sz = PROC_SHM_SIZE, // Get from clnt manager. NOT IN MONGO
        .proc_be_off = 0 // get from clnt manager - start offset. NOT IN MONGO
    };

    // mmap pci bar
    char *pci_path = _get_pci_path_clnt();
    assert(pci_path != NULL);

    int _pci_fd = open(pci_path, O_RDWR);
    assert(_pci_fd != -1);

    ivshmem_ctx->shm_mmap = mmap(NULL, _args.proc_be_sz, PROT_READ | PROT_WRITE, MAP_SHARED, _pci_fd, _args.proc_be_off);
    assert(ivshmem_ctx->shm_mmap != MAP_FAILED);
    close(_pci_fd);

    memcpy(ivshmem_ctx->shm_be_path, _args.f_be, strlen(_args.f_be) + 1);
    // ivshmem_ctx->shm_be_path = _args.f_be;
    ivshmem_ctx->shm_proc_start = _args.proc_be_off;
    ivshmem_ctx->shm_proc_size = _args.proc_be_sz;
    ivshmem_ctx->svc_args = _args;

    init_ivshmem_areas_clnt(ivshmem_ctx);
    
    //LOGE(LOG_DEBUG, "created ivshmem ctx\n");
    printf("clnt ivshmem ctx created. mmap VA: %p\n", ivshmem_ctx->shm_mmap);

    // increment be_off in client manager.
    _clnt_mgr_update_shm(ivshmem_ctx->pid);

}

void init_ivshmem_areas_clnt(ivshmem_clnt_ctx *ctx) {
    // write to [sz/2, sz)
    ctx->write_to.max_size = ctx->shm_proc_size  / 2;
    ctx->write_to.avail_size = ctx->shm_proc_size  / 2;
    ctx->write_to.avail_offset = ctx->shm_proc_start + PROC_WRITE_START_OFFSET_CLNT;

    // read from [0, sz/2)
    ctx->read_from.max_size = ctx->shm_proc_size  / 2;
    ctx->read_from.avail_size = ctx->shm_proc_size  / 2;
    ctx->read_from.avail_offset = ctx->shm_proc_start + PROC_READ_START_OFFSET_CLNT;
}

uintptr_t shm_get_writeaddr_clnt(ivshmem_clnt_ctx *ctx) {
    uintptr_t shm_va = (uintptr_t)ctx->shm_mmap;

    return (shm_va + PROC_WRITE_START_OFFSET_CLNT);
}

uintptr_t shm_get_readaddr_clnt(ivshmem_clnt_ctx *ctx) {
    uintptr_t shm_va = (uintptr_t)ctx->shm_mmap;

    return (shm_va + PROC_READ_START_OFFSET_CLNT);
}

off_t shm_get_write_area_offset(size_t sz) {
    // default, write to start of write_to area
    return (off_t)0;
}

off_t shm_get_read_area_offset(size_t sz) {
    // default, read from start of read_from area
    return (off_t)0;
}

int check_shm_limits(_ivshmem_area *area, int size) {
    if (size < area->max_size) {
        return 1;
    } else {
        //printf("Large memcpy, chunking...\n");
        return 0;
    }
}

#define MAX_LINE_LENGTH 4096
char *_get_pci_path_clnt() {
    FILE *pipe = popen("lspci -D | grep 'RAM memory: Red Hat, Inc. Inter-VM shared memory'", "r");
    char buffer[MAX_LINE_LENGTH];
    char *pci_path = NULL;

    if (pipe && fgets(buffer, sizeof(buffer), pipe)) {
        // Locate the first space in the output to isolate the PCI identifier
        char *space_pos = strchr(buffer, ' ');
        if (space_pos) {
            *space_pos = '\0'; // Truncate the string at the first space

            // Allocate memory for the full path
            pci_path = malloc(strlen("/sys/bus/pci/devices/") + strlen(buffer) + strlen("/resource2") + 1);
            if (pci_path) {
                sprintf(pci_path, "/sys/bus/pci/devices/%s/resource2", buffer);
            }
        }
    }
    
    pclose(pipe);
    return pci_path;
}

// clnt manager has a counter for last-used shm offset.
// Have a thread on clnt manager listenining for GET_SHM from UDS
// On getting this req, return proc size, be_offset to library.
ivshmem_setup_desc _clnt_mgr_get_shm(int clnt_pid, char *shm_be_path) {
    // send uds msg to client manager

    // recv string response

    // parse string to ivshmem_setup_desc

    // return _args struct
    ;
}

void _clnt_mgr_update_shm(int clnt_pid) {
    ;
}