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

resource_map_item* resource_map_get(resource_map* map, void* addr) {
    return &(map->list[(uint64_t)addr - OFFSET]);
}

uint8_t resource_map_contains(resource_map* map, void* addr) {
    LOGE(LOG_DEBUG, "Checking if %p is in resource map", addr);
    LOGE(LOG_DEBUG, "Tail idx: %lu", map->tail_idx);
    LOGE(LOG_DEBUG, "Free ptr idx: %lu", map->free_ptr_idx);
    LOGE(LOG_DEBUG, "Translated offset: %lu", addr - OFFSET);
    return (uint64_t)addr - OFFSET < map->tail_idx && map->list[(uint64_t)addr - OFFSET].present;
}

int resource_map_add(resource_map* map, void* orig_addr, void *args, void **new_addr) {
    if (map->free_ptr_idx >= map->length && map->tail_idx >= map->length) {
        resource_map_item *new_list = (resource_map_item*)realloc(map->list, sizeof(resource_map_item) * map->length * 2);
        if (new_list == NULL) {
            return -1;
        }
        map->list = new_list;
        map->length *= 2;
    }
    if (map->tail_idx == map->free_ptr_idx) {
        map->list[map->tail_idx].mapped_addr = orig_addr;
        map->list[map->tail_idx].args = args;
        map->list[map->tail_idx].present = 1;
        *new_addr = (void*)(map->tail_idx + OFFSET);
        map->tail_idx++;
        map->free_ptr_idx++;
    }
    else {
        uint64_t alloc_idx = map->free_ptr_idx;
        map->free_ptr_idx = (uint64_t)map->list[map->free_ptr_idx].mapped_addr;
        map->list[alloc_idx].mapped_addr = (void*)orig_addr;
        map->list[alloc_idx].args = args;
        map->list[alloc_idx].present = 1;
        *new_addr = (void*)(alloc_idx + OFFSET);
    }
    return 0;
}

void resource_map_unset(resource_map* map, void* mapped_addr) {
    map->list[(uint64_t)mapped_addr].mapped_addr = (void*)map->free_ptr_idx;
    map->list[(uint64_t)mapped_addr].present = 0;
    free(map->list[(uint64_t)mapped_addr].args);
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

    for (uint64_t i = map->free_ptr_idx; i < map->tail_idx; i = (uint64_t)map->list[i].mapped_addr) {
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

    return iter;
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
    iter->current_idx++;
    if (nextSetBit(iter->allocated, &(iter->current_idx))) {
        return iter->current_idx;
    }
    return 0;
}