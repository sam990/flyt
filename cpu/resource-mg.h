#ifndef _RESOURCE_MG_H_
#define _RESOURCE_MG_H_

#include "rpc/types.h"
#include "list.h"

typedef struct resource_mg_map_elem_t {
    void* client_address;
    void* cuda_address;
} resource_mg_map_elem;

typedef struct resource_mg_t {
    /* Restored resources where client address != cuda address
     * are stored here. This is a sorted list, enabling binary searching.
     * It contains elements of type resource_mg_map_elem
     */
    list map_res;
    /* During this run created resources where we use actual addresses on
     * the client side. This is an unordered list. We never have to search
     * this though. It containts elements of type void*.
     */
    list new_res;
    int bypass;
    pthread_mutex_t mutex;
} resource_mg;


//Runtime API RMs
resource_mg rm_events;
resource_mg rm_arrays;
resource_mg rm_kernels;

//Other RMs
resource_mg rm_cusolver;
resource_mg rm_cublas;
resource_mg rm_cublaslt;


//CUDNN RMs
resource_mg rm_cudnn;
resource_mg rm_cudnn_tensors;
resource_mg rm_cudnn_filters;
resource_mg rm_cudnn_tensortransform;
resource_mg rm_cudnn_poolings;
resource_mg rm_cudnn_activations;
resource_mg rm_cudnn_lrns;
resource_mg rm_cudnn_convs;
resource_mg rm_cudnn_backendds;


/** initializes the resource manager
 *
 * @bypass: if unequal to zero, searches for resources
 * will be bypassed, reducing the overhead. This is useful
 * for the original launch of an application as resources still
 * use their original pointers
 * @return 0 on success
 **/
int resource_mg_init(resource_mg *mg, int bypass);
int resource_mg_init_capacity(resource_mg *mg, int bypass, size_t capacity);
void resource_mg_free(resource_mg *mg);

int resource_mg_add_sorted(resource_mg *mg, void* client_address, void* cuda_address);
int resource_mg_create(resource_mg *mg, void* cuda_address);

int resource_mg_get(resource_mg *mg, void* client_address, void** cuda_address);

int resource_mg_get_element_at(resource_mg *mg, bool_t new_res, size_t at, void** element);

void *resource_mg_get_default(resource_mg *mg, void *client_address,
                              void *default_val);
void *resource_mg_get_or_null(resource_mg *mg, void *client_address);

void resource_mg_print(resource_mg *mg);

int resource_mg_remove(resource_mg *mg, void* client_address);

#endif //_RESOURCE_MG_H_
