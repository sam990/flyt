#ifndef __FLYT_CPU_RESOURCE_MAP_H__
#define __FLYT_CPU_RESOURCE_MAP_H__

#include <stdint.h>

typedef struct __resource_map_item {
    void *mapped_addr;
    void *args;
    uint8_t present;
} resource_map_item;

typedef struct __resource_map {
    resource_map_item *list;
    uint64_t length;
    uint64_t free_ptr_idx;
    uint64_t tail_idx;
} resource_map;

typedef struct __resource_map_iter {
    resource_map *map;
    uint64_t current_idx;
} resource_map_iter;

resource_map *init_resource_map(uint64_t init_length);

void free_resource_map(resource_map *map);

void *resource_map_addr_from_index(uint64_t idx);

uint64_t resource_map_index_from_addr(void *addr);

resource_map_item* resource_map_get(resource_map *map, void *mapped_addr);

void *resource_map_get_addr(resource_map *map, void *addr);

void *resource_map_get_addr_default(resource_map *map, void *addr,
                                    void *default_addr);

int resource_map_update_addr_idx(resource_map *map, uint64_t idx,
                                 void *new_addr);

uint8_t resource_map_contains(resource_map *map, void *addr);

int resource_map_add(resource_map *map, void *orig_addr, void* args, void **mapped_addr);

void resource_map_unset(resource_map *map, void *mapped_addr);

resource_map_iter *resource_map_init_iter(resource_map *map);

void resource_map_free_iter(resource_map_iter *iter);

uint64_t resource_map_iter_next(resource_map_iter *iter);


#endif // __FLYT_CPU_RESOURCE_MAP_H__

