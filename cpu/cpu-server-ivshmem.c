/**
 * FILE: cpu-server-ivshmem.c
 * --------------------------
 * Implements server-side shared-memory API
 * Thread safe
 */

#include "cpu-server-ivshmem.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>

ivshmem_svc_ctx *init_ivshmem_svc(ivshmem_setup_desc args_from_clnt) {
    // malloc
    ivshmem_svc_ctx *_ctx = malloc(sizeof(ivshmem_svc_ctx));
    assert(_ctx != NULL);

    _ctx->shm_enabled = args_from_clnt.iv_enable;
    _ctx->shm_proc_start = args_from_clnt.proc_be_off;
    _ctx->shm_proc_size = args_from_clnt.proc_be_sz;
    
    memcpy(_ctx->shm_be_path, args_from_clnt.f_be, strlen(args_from_clnt.f_be) + 1);

    // mmap
    int _be_fd = open(args_from_clnt.f_be, O_RDWR);
    assert(_be_fd != -1);

    _ctx->shm_mmap = mmap(NULL, args_from_clnt.proc_be_sz, PROT_READ | PROT_WRITE, MAP_SHARED, _be_fd, args_from_clnt.proc_be_off);
    assert(_ctx->shm_mmap != MAP_FAILED);
    close(_be_fd);

    // init ivshmem areas
    init_ivshmem_areas_svc(_ctx);

    printf("svc ivshmem ctx created. mmap VA: %p\n", _ctx->shm_mmap);

    return _ctx;
}

void init_ivshmem_areas_svc(ivshmem_svc_ctx *ctx) {
    // read from [sz/2, sz)
    ctx->write_to.max_size = ctx->shm_proc_size  / 2;
    ctx->read_from.avail_size = ctx->shm_proc_size / 2;
    ctx->read_from.avail_offset = ctx->shm_proc_start + ctx->shm_proc_size/2;

    // write to [0, sz/2)
    #define PROC_WRITE_START_OFFSET_SVC 0
    ctx->read_from.max_size = ctx->shm_proc_size  / 2;
    ctx->write_to.avail_size = ctx->shm_proc_size / 2;
    ctx->write_to.avail_offset = ctx->shm_proc_start + PROC_WRITE_START_OFFSET_SVC;
}

// cast to void * at runtime
uintptr_t shm_get_readaddr_svc(ivshmem_svc_ctx *_ctx) {
    uintptr_t shm_va = (uintptr_t)_ctx->shm_mmap;

    return (shm_va + (_ctx->shm_proc_size / 2));
}

uintptr_t shm_get_writeaddr_svc(ivshmem_svc_ctx *_ctx) {
    uintptr_t shm_va = (uintptr_t)_ctx->shm_mmap;

    return (shm_va + PROC_WRITE_START_OFFSET_SVC);
}