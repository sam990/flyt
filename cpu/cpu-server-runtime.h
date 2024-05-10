#ifndef _CPU_SERVER_RUNTIME_H_
#define _CPU_SERVER_RUNTIME_H_

int server_runtime_init(int restore, int gpu_id);
int server_runtime_deinit(void);
int server_runtime_checkpoint(const char *path, int dump_memory, unsigned long prog, unsigned long vers);
int server_runtime_restore(const char *path);


enum MEM_ALLOC_TYPE {
    MEM_ALLOC_TYPE_DEFAULT = 0,
};

typedef struct __mem_alloc_args {
    enum MEM_ALLOC_TYPE type;
    long long arg1;
    long long arg2;
    long long arg3;
    long long arg4;
    long long arg5;
    long long arg6; 
} mem_alloc_args_t;

#endif //_CPU_SERVER_RUNTIME_H_
