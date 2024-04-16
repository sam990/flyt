#include "cpu_rpc_prot.h"
#include "cpu-server.h"
#include "log.h"

#include <stdlib.h>
#include <stdint.h>

int main(int argc, char** argv)
{
    if (argc == 5) {
        uint64_t vers, memory;
        uint32_t gpu_id, num_sm_cores;
        if (sscanf(argv[1], "%lu", &vers) != 1) {
            printf("version string could not be converted to number\n");
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
        if (sscanf(argv[2], "%u", &gpu_id) != 1) {
            printf("gpu_id string could not be converted to number\n");
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
        if (sscanf(argv[3], "%u", &num_sm_cores) != 1) {
            printf("num_sm_cores string could not be converted to number\n");
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
        if (sscanf(argv[4], "%lu", &memory) != 1) {
            printf("memory string could not be converted to number\n");
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
    } 
    else {
        printf("usage: %s\n", argv[0]);
    }
    return 0;
}
