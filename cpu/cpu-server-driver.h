#ifndef _CPU_SERVER_DRIVER_H_
#define _CPU_SERVER_DRIVER_H_

#include <cuda.h>
#include "cpu-server-client-mgr.h"

int server_driver_init(int restore);
int server_driver_deinit(void);
//int server_driver_checkpoint(const char *path, int dump_memory, unsigned long prog, unsigned long vers);
//int server_driver_restore(const char *path);

int server_driver_reload_modules_data(cricket_client *client);

int server_driver_ctx_state_restore(CUcontext ctx);

int server_driver_ctx_state_restore_ckp(void);

#endif //_CPU_SERVER_DRIVER_H_
