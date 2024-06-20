#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.h"
#include "resource-map.h"

#define OFFSET 0x00ffffffffffffffull

resource_map* init_resource_map(uint64_t init_length) {
    resource_map* map = (resource_map*)malloc(sizeof(resource_map));
    if (map == NULL) {
        return NULL;
    }
    map->list = (resource_map_item*)malloc(sizeof(resource_map_item) * init_length);
    if (map->list == NULL) {
        free(map);
        return NULL;
    }
    map->length = init_length;
    map->free_ptr_idx = 1;
    map->tail_idx = 1;
    return map;
}

void free_resource_map(resource_map* map) {
    if (map == NULL) {
        return;
    }
    if (map->list != NULL) {
        free(map->list);
    }
    free(map);
}

void* resource_map_addr_from_index(uint64_t idx) {
    return (void*)(idx + OFFSET);
}

uint64_t resource_map_index_from_addr(void* addr) {
    return (uint64_t)addr - OFFSET;
}

resource_map_item* resource_map_get(resource_map* map, void* addr) {
    return &(map->list[(uint64_t)addr - OFFSET]);
}

void* resource_map_get_addr(resource_map* map, void *addr) {
    if (resource_map_contains(map, addr)) {
        return resource_map_get(map, addr)->mapped_addr;
    }
    return NULL;
}

void* resource_map_get_addr_default(resource_map* map, void *addr, void* default_addr) {
    if (resource_map_contains(map, addr)) {
        return resource_map_get(map, addr)->mapped_addr;
    }
    return default_addr;
}

int resource_map_update_addr_idx(resource_map* map, uint64_t idx, void* new_addr) {
    if (!resource_map_contains(map, resource_map_addr_from_index(idx))) {
        return -1;
    }
    map->list[idx].mapped_addr = new_addr;
    return 0;
}

uint8_t resource_map_contains(resource_map* map, void* addr) {
    return (uint64_t)addr - OFFSET < map->tail_idx && map->list[(uint64_t)addr - OFFSET].present;
}

int resource_map_add(resource_map* map, void* mapped_addr, void *args, void **client_addr) {
    if (map->free_ptr_idx >= map->length && map->tail_idx >= map->length) {
        resource_map_item *new_list = (resource_map_item*)realloc(map->list, sizeof(resource_map_item) * map->length * 2);
        if (new_list == NULL) {
            return -1;
        }
        map->list = new_list;
        map->length *= 2;
    }
    if (map->tail_idx == map->free_ptr_idx) {
        map->list[map->tail_idx].mapped_addr = mapped_addr;
        map->list[map->tail_idx].args = args;
        map->list[map->tail_idx].present = 1;
        *client_addr = (void*)(map->tail_idx + OFFSET);
        map->tail_idx++;
        map->free_ptr_idx++;
    }
    else {
        uint64_t alloc_idx = map->free_ptr_idx;
        map->free_ptr_idx = (uint64_t)map->list[map->free_ptr_idx].mapped_addr;
        map->list[alloc_idx].mapped_addr = (void*)mapped_addr;
        map->list[alloc_idx].args = args;
        map->list[alloc_idx].present = 1;
        *client_addr = (void*)(alloc_idx + OFFSET);
    }
    return 0;
}

void resource_map_unset(resource_map* map, void* client_addr) {
    uint64_t idx = (uint64_t)client_addr - OFFSET;

    map->list[idx].mapped_addr = (void*)map->free_ptr_idx;
    map->list[idx].present = 0;
    free(map->list[idx].args);
    map->free_ptr_idx = idx;
}

resource_map_iter* resource_map_init_iter(resource_map* map) {

    if (map == NULL) {
        return NULL;
    }

    resource_map_iter* iter = (resource_map_iter*)malloc(sizeof(resource_map_iter));
    if (iter == NULL) {
        return NULL;
    }
    iter->map = map;
    iter->current_idx = 0;

    return iter;
}

void resource_map_free_iter(resource_map_iter* iter) {
    if (iter == NULL) {
        return;
    }
    free(iter);
}

/**
 * Returns the index of items in resource map that are used
*/
uint64_t resource_map_iter_next(resource_map_iter* iter) {
    if (iter == NULL) {
        return 0;
    }
    if (iter->current_idx >= iter->map->tail_idx) {
        return 0;
    }
    while (iter->current_idx < iter->map->tail_idx && iter->map->list[iter->current_idx].present == 0) {
        iter->current_idx++;
    }
    
    return iter->current_idx >= iter->map->tail_idx ? 0 : iter->current_idx++;
}