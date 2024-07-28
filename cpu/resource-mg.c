#include "resource-mg.h"
#include "list.h"
#include "log.h"


int resource_mg_init(resource_mg *mg, int bypass)
{
    int ret = 0;
    if ((ret = list_init(&mg->new_res, sizeof(void*))) != 0) {
        LOGE(LOG_ERROR, "error initializing new_res list");
        goto out;
    }
    if (bypass == 0) {
        if ((ret = list_init(&mg->map_res, sizeof(resource_mg_map_elem))) != 0) {
            LOGE(LOG_ERROR, "error initializing map_res list");
            goto out;
        }
    }
    mg->bypass = bypass;
 out:
    return ret;
}

int resource_mg_init_capacity(resource_mg *mg, int bypass, size_t capacity)
{
    int ret = 0;
    if ((ret = list_init_capacity(&mg->new_res, sizeof(void*), capacity)) != 0) {
        LOGE(LOG_ERROR, "error initializing new_res list");
        goto out;
    }
    if (bypass == 0) {
        if ((ret = list_init_capacity(&mg->map_res, sizeof(resource_mg_map_elem), capacity)) != 0) {
            LOGE(LOG_ERROR, "error initializing map_res list");
            goto out;
        }
    }
    mg->bypass = bypass;
 out:
    return ret;
}

void resource_mg_free(resource_mg *mg)
{
    list_free(&mg->new_res);
    if (mg->bypass == 0) {
        list_free(&mg->map_res);
    }
}

int resource_mg_create(resource_mg *mg, void *cuda_address)
{
    if (list_append_copy(&mg->new_res, &cuda_address) != 0) {
        LOGE(LOG_ERROR, "failed to append to new_res");
        return 1;
    }
    return 0;
}

static int resource_mg_search_map(resource_mg *mg, void *client_address, void **cuda_address)
{
    size_t start = 0;
    size_t end;
    size_t mid;
    resource_mg_map_elem *mid_elem;
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return -1;
    }
    if (mg->map_res.length <= 0) {
        LOGE(LOG_DEBUG, "no resources in map_res");
        return -1;
    }
    end = mg->map_res.length-1;
    
    while (end >= start) {
        mid = start + (end-start)/2;
        mid_elem = list_get(&mg->map_res, mid);
        if (mid_elem == NULL) {
            LOG(LOG_ERROR, "list state of map_res is inconsistent");
            return -1;
        }

        if (mid_elem->client_address > client_address) {
            end = mid-1;
            if (mid == 0) {
                break;
            }
        } else if (mid_elem->client_address < client_address) {
            start = mid+1;
        } else /*if (mid_elem->client_address == client_address)*/ {
            *cuda_address = mid_elem->cuda_address;
            return 0;
        }
    }
    LOGE(LOG_DEBUG, "no find: %p", client_address);
    return -1;
}

void resource_mg_print(resource_mg *mg)
{
    size_t i;
    resource_mg_map_elem *elem;
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return;
    }
    LOG(LOG_DEBUG, "new_res:");
    for (i = 0; i < mg->new_res.length; i++) {
        LOG(LOG_DEBUG, "%p", *(void**)list_get(&mg->new_res, i));
    }
    if (mg->bypass == 0) {
        LOG(LOG_DEBUG, "map_res:");
        for (i = 0; i < mg->map_res.length; i++) {
            elem = list_get(&mg->map_res, i);
            LOG(LOG_DEBUG, "%p -> %p", elem->client_address, elem->cuda_address);
        }
    }
}

int resource_mg_get(resource_mg *mg, void* client_address, void** cuda_address)
{
    if (mg->bypass) {
        *cuda_address = client_address;
        return 0;
    }
    return resource_mg_search_map(mg, client_address, cuda_address);
}


void* resource_mg_get_default(resource_mg *mg, void* client_address, void* default_val)
{
    void *cuda_address;
    if (mg->bypass) {
        return client_address;
    } else {
        if (resource_mg_search_map(mg, client_address, &cuda_address) == 0) {
            return cuda_address;
        }
        else {
            return default_val;
        }
    }
}

void *resource_mg_get_or_null(resource_mg *mg, void *client_address) {
    return resource_mg_get_default(mg, client_address, NULL);
}

#include <stdio.h>
int resource_mg_add_sorted(resource_mg *mg, void* client_address, void* cuda_address)
{
    ssize_t start = 0;
    ssize_t end = mg->map_res.length-1;
    ssize_t mid;
    struct resource_mg_map_elem_t new_elem = {.client_address = client_address,
                                              .cuda_address = cuda_address};
    resource_mg_map_elem *mid_elem;
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return 1;
    }
    if (mg->bypass) {
        LOGE(LOG_ERROR, "cannot add to bypassed resource manager");
        return 1;
    }
    if (mg->map_res.length == 0) {
        return list_append_copy(&mg->map_res, &new_elem);
    }
    end = mg->map_res.length-1;
    
    while (end >= start) {
        mid = start + (end-start)/2;
        mid_elem = list_get(&mg->map_res, mid);
        if (mid_elem == NULL) {
            LOG(LOG_ERROR, "list state of map_res is inconsistent");
            return 1;
        }

        if (mid_elem->client_address > client_address) {
            end = mid-1;
            if (mid == 0) {
                break;
            }
        } else if (mid_elem->client_address < client_address) {
            start = mid+1;
        } else /*if (mid_elem->client_address == client_address)*/ {
            LOGE(LOG_WARNING, "duplicate resource! The first resource will be overwritten");
            mid_elem->cuda_address = cuda_address;
            return 0;
        }
    }
    if (end < 0LL) {
        end = 0;
    }
    resource_mg_map_elem *end_elem = list_get(&mg->map_res, end);
    if (end_elem->client_address < client_address) {
        end++;
    }
    return list_insert(&mg->map_res, end, &new_elem);
}


int resource_mg_remove(resource_mg *mg, void* client_address)
{
    size_t start = 0;
    size_t end;
    size_t mid;
    resource_mg_map_elem *mid_elem;
    if (mg == NULL) {
        LOGE(LOG_ERROR, "resource manager mg is NULL");
        return 1;
    }
    if (mg->bypass) {
        LOGE(LOG_ERROR, "cannot remove from bypassed resource manager");
        return 1;
    }
    if (mg->map_res.length == 0) {
        return 0;
    }
    end = mg->map_res.length-1;
    
    while (end >= start) {
        mid = start + (end-start)/2;
        mid_elem = list_get(&mg->map_res, mid);
        if (mid_elem == NULL) {
            LOG(LOG_ERROR, "list state of map_res is inconsistent");
            return 1;
        }

        if (mid_elem->client_address > client_address) {
            end = mid-1;
            if (mid == 0) {
                break;
            }
        } else if (mid_elem->client_address < client_address) {
            start = mid+1;
        } else /*if (mid_elem->client_address == client_address)*/ {
            return list_rm(&mg->map_res, mid);
        }
    }
    return 0;
}
