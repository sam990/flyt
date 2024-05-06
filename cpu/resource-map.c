#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "resource-map.h"

resource_map* init_resource_map(u_int64_t init_length) {
    resource_map* map = (resource_map*)malloc(sizeof(resource_map));
    if (map == NULL) {
        return NULL;
    }
    map->list = (u_int64_t*)malloc(sizeof(u_int64_t) * init_length);
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

uint64_t get_resource(resource_map* map, void* mapped_addr) {
    return map->list[(uint64_t)mapped_addr];
}

int set_resource(resource_map* map, void* orig_addr, void **mapped_addr) {
    if (map->free_ptr_idx >= map->length && map->tail_idx >= map->length) {
        uint64_t *new_list = (uint64_t*)realloc(map->list, sizeof(uint64_t) * map->length * 2);
        if (new_list == NULL) {
            return -1;
        }
        map->list = new_list;
        map->length *= 2;
    }
    if (map->tail_idx == map->free_ptr_idx) {
        map->list[map->tail_idx] = (uint64_t)orig_addr;
        *mapped_addr = (void*)map->tail_idx;
        map->tail_idx++;
        map->free_ptr_idx++;
    }
    else {
        uint64_t alloc_idx = map->free_ptr_idx;
        map->free_ptr_idx = map->list[map->free_ptr_idx];
        map->list[alloc_idx] = (uint64_t)orig_addr;
        *mapped_addr = (void*)alloc_idx;
    }
    return 0;
}

void unset_resource(resource_map* map, void* mapped_addr) {
    map->list[(uint64_t)mapped_addr] = map->free_ptr_idx;
    map->free_ptr_idx = (uint64_t)mapped_addr;
}

resource_map_iter* resource_map_init_iter(resource_map* map) {

    if (map == NULL) {
        return NULL;
    }

    bitset_t *allocated =  bitset_create_with_capacity(map->tail_idx);
    if (allocated == NULL) {
        return NULL;
    }

    bitset_t* free_ptr = bitset_create_with_capacity(map->tail_idx);
    if (free_ptr == NULL) {
        bitset_free(allocated);
        return NULL;
    }

    for (uint64_t i = map->free_ptr_idx; i < map->tail_idx; i = map->list[i]) {
        bitset_set(free_ptr, i);
    }

    for (uint64_t i = 1; i < map->tail_idx; i++) {
        if (bitset_get(free_ptr, i) == false) {
            bitset_set(allocated, i);
        }
    }

    resource_map_iter* iter = (resource_map_iter*)malloc(sizeof(resource_map_iter));
    if (iter == NULL) {
        bitset_free(allocated);
        bitset_free(free_ptr);
        return NULL;
    }

    iter->map = map;
    iter->allocated = allocated;
    iter->current_idx = 0;

    return 0;
}

void resource_map_free_iter(resource_map_iter* iter) {
    if (iter == NULL) {
        return;
    }
    if (iter->allocated != NULL) {
        bitset_free(iter->allocated);
    }
    free(iter);
}

uint64_t resource_map_iter_next(resource_map_iter* iter) {
    if (iter == NULL) {
        return 0;
    }
    if (iter->current_idx >= iter->map->tail_idx) {
        return 0;
    }
    iter->current_idx++;
    if (nextSetBit(iter->allocated, &(iter->current_idx))) {
        return iter->current_idx;
    }
    return 0;
}