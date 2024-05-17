#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>

#include "cpu-server-client-mgr.h"
#include "resource-map.h"
#include "resource-mg.h"
#include "cpu_rpc_prot.h"
#include "cpu-server-runtime.h"
#include "log.h"
#include "list.h"
#include "cpu-server-resource-controller.h"
#include "cpu-server-driver.h"

#include "flyt-cr.h"
#include <dirent.h>

int dump_memory(char *filename, resource_map *gpu_mem) {
    FILE *fp = fopen(filename, "wb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for writing", filename);
        return -1;
    }

    fwrite(gpu_mem, sizeof(resource_map), 1, fp);
    fwrite(gpu_mem->list, sizeof(resource_map_item), gpu_mem->length, fp);

    for (uint64_t i = 1; i < gpu_mem->tail_idx; i++) {
        if (gpu_mem->list[i].present) {
            
            mem_alloc_args_t *alloc_args = (mem_alloc_args_t *)gpu_mem->list[i].args;
            // get the data from the device
            uint8_t *data = (uint8_t *)malloc(alloc_args->size);
            if (data == NULL) {
                LOGE(LOG_ERROR, "Failed to allocate memory for data");
                fclose(fp);
                return -1;
            }

            if (cudaMemcpy(data, gpu_mem->list[i].mapped_addr, alloc_args->size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                LOGE(LOG_ERROR, "Failed to copy data from device to host");
                free(data);
                fclose(fp);
                return -1;
            }
            fwrite(gpu_mem->list[i].args, sizeof(mem_alloc_args_t), 1, fp);
            fwrite(data, alloc_args->size, 1, fp);
            free(data);
        }
    }
    fclose(fp);
    return 0;
}


int dump_modules(char *filename, resource_mg* modules) {
    
    FILE *fp = fopen(filename, "wb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for writing", filename);
        return -1;
    }

    for (uint64_t i = 0; i < modules->map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&modules->map_res, i);

        addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
        rpc_elf_load_1_argument * data = (rpc_elf_load_1_argument *)pair->reg_data;

        LOGE(LOG_DEBUG, "Dumping module %p, len: %u", data->arg2, data->arg1.mem_data_len);

        fwrite(data, sizeof(rpc_elf_load_1_argument), 1, fp);
        fwrite(data->arg1.mem_data_val, data->arg1.mem_data_len, 1, fp);
    }

    fclose(fp);
    return 0;
}


int dump_functions(char *filename, resource_mg* functions) {
    
    FILE *fp = fopen(filename, "wb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for writing", filename);
        return -1;
    }

    for (uint64_t i = 0; i < functions->map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&functions->map_res, i);

        addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
        rpc_register_function_1_argument * data = (rpc_register_function_1_argument *)pair->reg_data;

        char arg3_str[128];
        char arg4_str[128];

        strncpy(arg3_str, data->arg3, 128);
        strncpy(arg4_str, data->arg4, 128);

        fwrite(data, sizeof(rpc_register_function_1_argument), 1, fp);
        fwrite(arg3_str, 128, 1, fp);
        fwrite(arg4_str, 128, 1, fp);
    }

    fclose(fp);
    return 0;
}

int dump_vars(char *filename, resource_mg *vars) {
        
    FILE *fp = fopen(filename, "wb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for writing", filename);
        return -1;
    }

    for (uint64_t i = 0; i < vars->map_res.length; i++) {
        resource_mg_map_elem *elem = list_get(&vars->map_res, i);
        void* client_addr = elem->client_address;

        addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
        rpc_register_var_1_argument * data = (rpc_register_var_1_argument *)pair->reg_data;

        size_t size = data->arg6;
        char *var_data = malloc(size);

        cudaMemcpyFromSymbol(var_data, pair->addr, size, 0, cudaMemcpyDeviceToHost);

        char arg4_str[128];

        strncpy(arg4_str, data->arg4, 128);

        fwrite(data, sizeof(rpc_register_var_1_argument), 1, fp);
        fwrite(arg4_str, 128, 1, fp);
        fwrite(var_data, size, 1, fp);

        free(var_data);
    }

    fclose(fp);
    return 0;
}


int flyt_create_checkpoint(char *basepath) {
    cudaDeviceSynchronize();

    if (access(basepath, F_OK) != 0 && mkdir(basepath, 0777) == -1) {
        LOGE(LOG_ERROR, "Error creating checkpoint directory %s: %s", basepath, strerror(errno));
        return -1;
    }

    cricket_client_iter iter = get_client_iter();
    cricket_client *client;

    while ((client = get_next_client(&iter)) != NULL) {
        char *client_path = malloc(strlen(basepath) + 32);
        if (client_path == NULL) {
            LOGE(LOG_ERROR, "Failed to allocate memory for client path");
            return -1;
        }

        sprintf(client_path, "%s/%d", basepath, client->pid);

        if (mkdir(client_path, 0777) == -1) {
            LOGE(LOG_ERROR, "Error creating client directory %s: %s", client_path, strerror(errno));
            free(client_path);
            return -1;
        }

        char *filename = malloc(strlen(client_path) + 32);
        if (filename == NULL) {
            LOGE(LOG_ERROR, "Failed to allocate memory for filename");
            free(client_path);
            return -1;
        }

        sprintf(filename, "%s/gpu_mem", client_path);
        if (dump_memory(filename, client->gpu_mem) != 0) {
            LOGE(LOG_ERROR, "Failed to dump memory for client %d", client->pid);
            free(client_path);
            free(filename);
            return -1;
        }

        sprintf(filename, "%s/modules", client_path);
        if (dump_modules(filename, &client->modules) != 0) {
            LOGE(LOG_ERROR, "Failed to dump modules for client %d", client->pid);
            free(client_path);
            free(filename);
            return -1;
        }

        sprintf(filename, "%s/functions", client_path);
        if (dump_functions(filename, &client->functions) != 0) {
            LOGE(LOG_ERROR, "Failed to dump functions for client %d", client->pid);
            free(client_path);
            free(filename);
            return -1;
        }

        sprintf(filename, "%s/vars", client_path);
        if (dump_vars(filename, &client->vars) != 0) {
            LOGE(LOG_ERROR, "Failed to dump vars for client %d", client->pid);
            free(client_path);
            free(filename);
            return -1;
        }

        free(client_path);
        free(filename);

    }

    return 0;
}

int flyt_restore_memory(char *memory_file, resource_map *gpu_mem) {
    FILE *fp = fopen(memory_file, "rb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for reading", memory_file);
        return -1;
    }


    size_t readsz;

    free(gpu_mem->list);

    readsz = fread(gpu_mem, sizeof(resource_map), 1, fp);

    if (readsz != 1) {
        fclose(fp);
        LOGE(LOG_ERROR, "Failed to read memory map from file");
        return -1;
    }

    gpu_mem->list = malloc(sizeof(resource_map_item) * gpu_mem->length);

    if (gpu_mem->list == NULL) {
        LOGE(LOG_ERROR, "Failed to allocate memory for memory map");
        fclose(fp);
        return -1;
    }

    readsz = fread(gpu_mem->list, sizeof(resource_map_item), gpu_mem->length, fp);

    if (readsz != gpu_mem->length) {
        fclose(fp);
        LOGE(LOG_ERROR, "Failed to read memory map from file");
        return -1;
    }

    for (uint64_t i = 1; i < gpu_mem->tail_idx; i++) {
        if (gpu_mem->list[i].present) {
            gpu_mem->list[i].args = malloc(sizeof(mem_alloc_args_t));
            
            if (gpu_mem->list[i].args == NULL) {
                LOGE(LOG_ERROR, "Failed to allocate memory for memory args");
                fclose(fp);
                return -1;
            }

            readsz = fread(gpu_mem->list[i].args, sizeof(mem_alloc_args_t), 1, fp);
            if (readsz != 1) {
                LOGE(LOG_ERROR, "Failed to read memory args from file");
                fclose(fp);
                return -1;
            }

            mem_alloc_args_t *alloc_args = (mem_alloc_args_t *)gpu_mem->list[i].args;
            // get the data from the device
            uint8_t *data = (uint8_t *)malloc(alloc_args->size);
            if (data == NULL) {
                LOGE(LOG_ERROR, "Failed to allocate memory for data");
                fclose(fp);
                return -1;
            }

            readsz = fread(data, alloc_args->size, 1, fp);

            if (readsz != 1) {
                free(data);
                fclose(fp);
                return -1;
            }

            if (cudaMalloc(&gpu_mem->list[i].mapped_addr, alloc_args->size) != cudaSuccess) {
                LOGE(LOG_ERROR, "Failed to allocate memory on device");
                free(data);
                fclose(fp);
                return -1;
            }

            if (cudaMemcpy(gpu_mem->list[i].mapped_addr, data, alloc_args->size, cudaMemcpyHostToDevice) != cudaSuccess) {
                LOGE(LOG_ERROR, "Failed to copy data to device");
                free(data);
                fclose(fp);
                return -1;
            }

            free(data);
        }
    }

    fclose(fp);
    return 0;
}


int flyt_restore_modules(char *modules_file, resource_mg *modules) {
    FILE *fp = fopen(modules_file, "rb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for reading", modules_file);
        return -1;
    }

    size_t readsz;

    while (!feof(fp)) {
        rpc_elf_load_1_argument data;
        readsz = fread(&data, sizeof(rpc_elf_load_1_argument), 1, fp);

        if (readsz != 1) {
            break;
        }

        LOGE(LOG_DEBUG, "Restoring module %p, len: %u", data.arg2, data.arg1.mem_data_len);

        char *mem_data = malloc(data.arg1.mem_data_len);
        if (mem_data == NULL) {
            LOGE(LOG_ERROR, "Failed to allocate memory for mem_data");
            fclose(fp);
            return -1;
        }

        readsz = fread(mem_data, data.arg1.mem_data_len, 1, fp);

        if (readsz != 1) {
            LOGE(LOG_ERROR, "Failed to read module mem_data from file");
            free(mem_data);
            fclose(fp);
            return -1;
        }

        rpc_elf_load_1_argument *data_ptr = malloc(sizeof(rpc_elf_load_1_argument));
        data_ptr->arg1.mem_data_val = mem_data;
        data_ptr->arg1.mem_data_len = data.arg1.mem_data_len;
        data_ptr->arg2 = data.arg2;

        addr_data_pair_t *pair = malloc(sizeof(addr_data_pair_t));
        pair->addr = NULL;
        pair->reg_data = data_ptr;

        resource_mg_add_sorted(modules, (void *)data_ptr->arg2, (void *)pair);
    }

    fclose(fp);
    return 0;
}

int flyt_restore_functions(char *functions_file, resource_mg *functions) {
    FILE *fp = fopen(functions_file, "rb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for reading", functions_file);
        return -1;
    }

    size_t readsz;

    while (!feof(fp)) {
        rpc_register_function_1_argument data;
        readsz = fread(&data, sizeof(rpc_register_function_1_argument), 1, fp);

        if (readsz != 1) {
            break;
        }

        char *arg3 = malloc(128);
        char *arg4 = malloc(128);

        if (arg3 == NULL || arg4 == NULL) {
            LOGE(LOG_ERROR, "Failed to allocate memory for arg3 or arg4");
            fclose(fp);
            return -1;
        }

        readsz = fread(arg3, 128, 1, fp);
        readsz += fread(arg4, 128, 1, fp);

        if (readsz != 2) {
            LOGE(LOG_ERROR, "Failed to read function args from file");
            free(arg3);
            free(arg4);
            fclose(fp);
            return -1;
        }

        rpc_register_function_1_argument *data_ptr = malloc(sizeof(rpc_register_function_1_argument));
        memcpy(data_ptr, &data, sizeof(rpc_register_function_1_argument));
        data_ptr->arg3 = arg3;
        data_ptr->arg4 = arg4;

        addr_data_pair_t *pair = malloc(sizeof(addr_data_pair_t));
        pair->addr = NULL;
        pair->reg_data = data_ptr;

        resource_mg_add_sorted(functions, (void *)data_ptr->arg2, (void *)pair);
    }

    fclose(fp);
    return 0;
}

int flyt_restore_vars(char *vars_file, resource_mg *vars) {
    FILE *fp = fopen(vars_file, "rb");

    if (fp == NULL) {
        LOGE(LOG_ERROR, "Failed to open file %s for reading", vars_file);
        return -1;
    }

    size_t readsz;

    while (!feof(fp)) {
        rpc_register_var_1_argument data;
        readsz = fread(&data, sizeof(rpc_register_var_1_argument), 1, fp);

        if (readsz != 1) {
            break;
        }

        char *arg4 = malloc(128);
        if (arg4 == NULL) {
            LOGE(LOG_ERROR, "Failed to allocate memory for arg4");
            fclose(fp);
            return -1;
        }

        readsz = fread(arg4, 128, 1, fp);

        if (readsz != 1) {
            LOGE(LOG_ERROR, "Failed to read var arg from file");
            free(arg4);
            fclose(fp);
            return -1;
        }

        rpc_register_var_1_argument *data_ptr = malloc(sizeof(rpc_register_var_1_argument));
        memcpy(data_ptr, &data, sizeof(rpc_register_var_1_argument));
        data_ptr->arg4 = arg4;

        addr_data_pair_t *pair = malloc(sizeof(addr_data_pair_t));
        pair->addr = NULL;
        pair->reg_data = data_ptr;

        resource_mg_add_sorted(vars, (void *)data_ptr->arg2, (void *)pair);
    }

    fclose(fp);
    return 0;
}


int flyt_restore_checkpoint(char *basepath) {
    // sleep(30);
    if (access(basepath, F_OK) == -1) {
        LOGE(LOG_ERROR, "Checkpoint directory %s does not exist", basepath);
        return -1;
    }

    // list all subdirs
    DIR *dir = opendir(basepath);
    if (dir == NULL) {
        LOGE(LOG_ERROR, "Failed to open checkpoint directory %s: %s", basepath, strerror(errno));
        return -1;
    }

    struct dirent *ent;

    // iterate over all subdirs
    for (ent = readdir(dir); ent != NULL; ent = readdir(dir)) {
        if (ent->d_type == DT_DIR) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                continue;
            }

            char *client_pid = ent->d_name;
            char *client_path = malloc(strlen(basepath) + strlen(client_pid) + 2);
            if (client_path == NULL) {
                LOGE(LOG_ERROR, "Failed to allocate memory for client path");
                return -1;
            }

            sprintf(client_path, "%s/%s", basepath, client_pid);
            int client_pid_int = atoi(client_pid);

            cricket_client *client = create_client(client_pid_int);

            if (client == NULL) {
                LOGE(LOG_ERROR, "Failed to create client for pid %d", client_pid_int);
                free(client_path);
                return -1;
            }

            LOGE(LOG_DEBUG, "Restoring checkpoint for client %d", client_pid_int);

            char *filename = malloc(strlen(client_path) + 32);
            if (filename == NULL) {
                LOGE(LOG_ERROR, "Failed to allocate memory for filename");
                free(client_path);
                remove_client_ptr(client);
                return -1;
            }

            int ret;

            sprintf(filename, "%s/modules", client_path);
            ret = flyt_restore_modules(filename, &client->modules);
            
            sprintf(filename, "%s/functions", client_path);
            ret |= flyt_restore_functions(filename, &client->functions);

            sprintf(filename, "%s/vars", client_path);
            ret |= flyt_restore_vars(filename, &client->vars);

            sprintf(filename, "%s/gpu_mem", client_path);
            ret |= flyt_restore_memory(filename, client->gpu_mem);

            if (ret != 0) {
                LOGE(LOG_ERROR, "Failed to restore checkpoint for client %d", client_pid_int);
                free(client_path);
                free(filename);
                remove_client_ptr(client);
                return -1;
            }

            LOGE(LOG_DEBUG, "Restored checkpoint data for client %d", client_pid_int);

            free(client_path);
            free(filename);

            add_restored_client(client);

        }

    }

    closedir(dir);

    // sleep(5);
    int ret = server_driver_ctx_state_restore_ckp();

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to restore context state");
        return -1;
    }

    cudaDeviceSynchronize();
    
    return 0;

}


