#include "cpu_rpc_prot.h"
#include "cpu-server.h"
#include "log.h"

#include <stdlib.h>
#include <stdint.h>

int main(int argc, char** argv)
{
    uint64_t vers, memory;
    uint32_t gpu_id, num_sm_cores;
    if (argc == 5) {
        if (sscanf(argv[1], "%lu", &vers) != 1) {
            printf("unique rpc version string could not be converted to number: %s\n", argv[1]);
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
        if (sscanf(argv[2], "%u", &gpu_id) != 1) {
            printf("gpu_id string could not be converted to number: %s\n", argv[2]);
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
        if (sscanf(argv[3], "%u", &num_sm_cores) != 1) {
            printf("num_sm_cores string could not be converted to number: %s\n", argv[3]);
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
        if (sscanf(argv[4], "%lu", &memory) != 1) {
            printf("memory string could not be converted to number: %s\n", argv[4]);
            printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
            return 1;
        }
    } 
    else {
        printf("usage: %s [unique rpc version] [gpu_id] [num_sm_cores] [memory]\n", argv[0]);
        return 1;
    }

    // Start the server
    cricket_main(RPC_CD_PROG, vers, gpu_id, num_sm_cores, memory);

    return 0;
}
