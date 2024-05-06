#ifndef __FLYT_CPU_RESOURCE_MAP_H__
#define __FLYT_CPU_RESOURCE_MAP_H__

#include <stdint.h>
#include "cbitset/bitset.h"

typedef struct __resource_map {
    uint64_t *list;
    uint64_t length;
    uint64_t free_ptr_idx;
    uint64_t tail_idx;
} resource_map;

typedef struct __resource_map_iter {
    resource_map *map;
    bitset_t *allocated;
    size_t current_idx;
} resource_map_iter;

resource_map *init_resource_map(u_int64_t init_length);

void free_resource_map(resource_map *map);

uint64_t get_resource(resource_map *map, void *mapped_addr);

int set_resource(resource_map *map, void *orig_addr, void **mapped_addr);

void unset_resource(resource_map *map, void *mapped_addr);

resource_map_iter *resource_map_init_iter(resource_map *map);

void resource_map_free_iter(resource_map_iter *iter);

uint64_t resource_map_iter_next(resource_map_iter *iter);


#endif // __FLYT_CPU_RESOURCE_MAP_H__

