#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda_runtime.h>

#include "log.h"
#include "resource-map.h"
#include "resource-mg.h"
#include "cpu-server-client-mgr.h"
#include "cpu-server-resource-controller.h"
#include "cpu_rpc_prot.h"
#include "cpu-server-dev-mem.h"

#define STR_DUMP_SIZE 128

static pthread_mutex_t client_mgr_mutex = PTHREAD_MUTEX_INITIALIZER;


static resource_mg pid_to_xp_fd;
static resource_mg xp_fd_to_client;
static resource_mg restored_clients;

// Metric measurement throughput
// #include <time.h>
static clock_t kernel_start;
static int kernel_launch_count = 0;
static int kernel_thread_count = 0;

#define MAX_CALLS 100

struct call_data {
    struct timeval start_time;
    struct timeval end_time;
    double resource_usage;
};

static struct call_data calls[MAX_CALLS];
static int call_count = 0, call_insert_index = 0, end_call_count = 0, end_call_index = 0;


// Metric measurement utilization
#include <cupti.h>
#define CHECK_CUPTI_ERROR(err, lock) \
    if (err != CUPTI_SUCCESS) { \
        const char *errstr; \
        cuptiGetResultString(err, &errstr); \
        LOGE(LOG_ERROR, "CUPTI Error: %s", errstr); \
	if (lock) \
          pthread_mutex_unlock(&client_mgr_mutex); \
        return; \
    }

// Check for CUDA errors
#define CHECK_CUDA_ERROR(err) \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        LOGE(LOG_ERROR, "CUDA Error %s", errStr); \
        return; \
    }

// Select the metrics
static const char *metricNames[CUPTI_NUM_METRICS] = {
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__cycles_active.avg",
        "sm__inst_executed.sum"
    };

static pthread_cond_t client_mgr_cond = PTHREAD_COND_INITIALIZER;
static int resetevents = 0;
static CUpti_EventGroupSets *eventGroupSets = NULL;
static CUpti_MetricID metricIDs[CUPTI_NUM_METRICS];
static uint64_t metricsValue[CUPTI_NUM_METRICS];
static uint64_t prevMetricsValue[CUPTI_NUM_METRICS];

void update_client_metric_values(CUdevice currentDevice, CUcontext currentCtx) {
	/*

    if(eventGroupSets == NULL) {
        LOGE(LOG_ERROR, "eventGroupSets not initialized");
	return;
    }

    // Read and process the metrics
    for (int i = 0; i < CUPTI_NUM_METRICS; i++) {
        CUpti_MetricValue metricValue;

        CUpti_EventGroupSet *eventGroupSet = &eventGroupSets->sets[i];

	for (size_t j = 0; j < eventGroupSet->numEventGroups; ++j) {
            CUpti_EventGroup eventGroup = eventGroupSet->eventGroups[j];
            CUpti_EventID *eventIDs;
            uint32_t numEvents;
            size_t bytesRead;

	    bytesRead = sizeof(numEvents);
            CHECK_CUPTI_ERROR( cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &bytesRead, &numEvents), 0);

            eventIDs = (CUpti_EventID *)malloc(numEvents * sizeof(CUpti_EventID));
	    if(eventIDs == NULL) {
		LOGE(LOG_ERROR, "not able to malloc eventIds ");
		pthread_mutex_unlock(&client_mgr_mutex);
		return;
	    }
	    bytesRead = numEvents * sizeof(CUpti_EventID);
            CHECK_CUPTI_ERROR( cuptiEventGroupGetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_EVENTS, &bytesRead, eventIDs), 0);

            uint64_t *eventValues = (uint64_t *)malloc(numEvents * sizeof(uint64_t));
	    if(eventValues == NULL) {
		LOGE(LOG_ERROR, "not able to malloc eventValues ");
		free(eventIDs);
		pthread_mutex_unlock(&client_mgr_mutex);
		return;
	    }
            bytesRead = numEvents * sizeof(uint64_t);
            CHECK_CUPTI_ERROR( cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE, eventIDs[0], &bytesRead, eventValues), 0);

            // Use event values to calculate the metric value
            CUpti_MetricValue metricValue;
            CHECK_CUPTI_ERROR( cuptiMetricGetValue(currentDevice, metricIDs[i], bytesRead, eventIDs, sizeof(eventIDs), eventValues, 0, &metricValue), 0);
	    metricsValue[i] = (uint64_t ) metricNames[i], metricValue.metricValueUint64;

	   free(eventIDs);
	   free(eventValues);
	}
    }
    */
}

void update_client_metric_utilization() {
    CUdevice currentDevice;
    CUcontext currentCtx = NULL;
    /*

    CHECK_CUDA_ERROR(cuCtxGetDevice(&currentDevice));
    CHECK_CUDA_ERROR(cuCtxGetCurrent(&currentCtx));

    pthread_mutex_lock(&client_mgr_mutex);
    if (eventGroupSets == NULL) {
	uint32_t version;
    	CHECK_CUPTI_ERROR(cuptiGetVersion(&version), 1);
    	LOGE(LOG_INFO, "CUPTI Version: %d\n", version);
	// fill in event Ids
	//CHECK_CUPTI_ERROR(cuptiSetEventCollectionMode(currentCtx, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS), 1);
	
	// Get metric IDs
	for (int i = 0; i < CUPTI_NUM_METRICS; i++) {
	    CHECK_CUPTI_ERROR( cuptiMetricGetIdFromName(currentDevice, metricNames[i], &metricIDs[i]), 1);
        }

	// Create event groups for these meterics
	CHECK_CUPTI_ERROR(cuptiMetricCreateEventGroupSets(currentCtx, sizeof(metricIDs), metricIDs, &eventGroupSets), 1);

	// Enable event groups in CONTINUOUS mode
        for (size_t i = 0; i < eventGroupSets->numSets; ++i) {
            CUpti_EventGroupSet *eventGroupSet = &eventGroupSets->sets[i];
            for (size_t j = 0; j < eventGroupSet->numEventGroups; ++j) {
                uint32_t enabled = 1;
                CHECK_CUPTI_ERROR( cuptiEventGroupSetAttribute(eventGroupSet->eventGroups[j], CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(uint32_t), &enabled), 1);
            }
            CHECK_CUPTI_ERROR( cuptiEventGroupSetEnable(eventGroupSet), 1);
        }
    }
    else {
        update_client_metric_values(currentDevice, currentCtx);

	if ((resetevents != 0) && (eventGroupSets != NULL)) {
	    // Disable and destroy existing event groups
	    for (size_t i = 0; i < eventGroupSets->numSets; ++i) {
		 CUpti_EventGroupSet *eventGroupSet = &eventGroupSets->sets[i];
		 CHECK_CUPTI_ERROR(cuptiEventGroupSetDisable(eventGroupSet), 1);
		 for (size_t j = 0; j < eventGroupSet->numEventGroups; ++j) {
		    CHECK_CUPTI_ERROR(cuptiEventGroupDestroy(eventGroupSet->eventGroups[j]), 1);
		 }
	    }
	    CHECK_CUPTI_ERROR(cuptiMetricDestroyEventGroupSets(eventGroupSets), 1);
	    eventGroupSets = NULL;
	    resetevents = 0;
	    for (int i = 0; i < CUPTI_NUM_METRICS; i++) {
		    prevMetricsValue[i] = metricsValue[i];
		    metricsValue[i] = 0;
	    }
	    pthread_cond_signal(&client_mgr_cond);
        }
    }

    pthread_mutex_unlock(&client_mgr_mutex);
    */

}

void get_client_metric_utilization(uint64_t *valueArray, int count) {
    int params;
    /*

    for (int i = 0; i< count; i++)
	    valueArray[i] = 0;

    pthread_mutex_lock(&client_mgr_mutex);

    if(eventGroupSets == NULL) {
        LOGE(LOG_ERROR, "eventGroupSets not initialized");
        pthread_mutex_unlock(&client_mgr_mutex);
	return;
    }

    // Notify to destory
    resetevents = 1;
    while(resetevents == 0) {
        pthread_cond_wait(&client_mgr_cond, &client_mgr_mutex);
    }

    // Copy values.
    params = count > CUPTI_NUM_METRICS? CUPTI_NUM_METRICS : count;
    // Read and process the metrics
    for (int i = 0; i < params; i++) {
	valueArray[i] = prevMetricsValue[i];
	prevMetricsValue[i] = 0;
    }
    pthread_mutex_unlock(&client_mgr_mutex);
    */
}

void update_client_metric_throughput(cricket_client *client, int threads) {

    pthread_mutex_lock(&client_mgr_mutex);

    if (threads) {
	    if (call_count < MAX_CALLS) {
		// Capture start time
		gettimeofday(&calls[call_count].start_time, NULL);
		// Store the resource usage at the start
		calls[call_count].resource_usage = (double) threads;
		call_count++;
	    }
	    else {
		// If the buffer is full, overwrite the oldest request
		gettimeofday(&calls[call_insert_index].start_time, NULL);
        	calls[call_insert_index].resource_usage = (double) threads;
        	call_insert_index = (call_insert_index + 1) % MAX_CALLS;
	    }
    }
    else {
	if (end_call_count < call_count) {
	     if	(end_call_count < MAX_CALLS) {
                // Capture end time
                gettimeofday(&calls[end_call_count].end_time, NULL);
                end_call_count++;
	     }
	     else {
                gettimeofday(&calls[end_call_index].end_time, NULL);
        	end_call_index = (end_call_index + 1) % MAX_CALLS;
	     }
        }
    }

    //LOGE(LOG_INFO, "Start count %d end count %d resource %d call count %d end_count %d end_coud %d", calls[call_count-1].start_time.tv_sec, calls[call_count -1].end_time.tv_sec, calls[call_count-1].resource_usage, call_count, end_call_count, end_call_count);

    /*
    if(kernel_launch_count == 0) {
	    kernel_start = clock(); //time timestamp
    }
    kernel_launch_count++;
    kernel_thread_count += threads;
    */
    pthread_mutex_unlock(&client_mgr_mutex);
}

// Function to compare two integers for qsort
int compare_doubles(const void *a, const void *b) {
    double diff = (*(double *)a - *(double *)b);
    
    if (diff < 0.0) return -1;  // a is less than b
    else if (diff > 0.0) return 1;  // a is greater than b
    else return 0;  // a and b are equal
}

// Function to calculate the 99th percentile from the requests
double compute_99th_percentile() {
    static double sorted_resource[MAX_CALLS];

    if (end_call_count == 0) {
        return 0;
    }

    for (int i = 0; i < end_call_count; i++) {
        int index = (end_call_index + i) % MAX_CALLS;
        sorted_resource[i] = calls[index].resource_usage;
    }

    // Sort the requests
    qsort(sorted_resource, end_call_count, sizeof(double), compare_doubles);

    // Calculate the index for the 99th percentile
    int idx = (int)((end_call_count - 1) * 0.99);
    double percentile_value = (double)sorted_resource[idx];

    return percentile_value;
}

#include <math.h>
void get_client_metric_throughput(uint32_t *avg_resource_usage, uint32_t *avg_function_rate, uint32_t *avg_latency) {

    struct cudaDeviceProp dev_prop;
    *avg_resource_usage = 0;
    *avg_function_rate = 0;
    *avg_latency = 0;
    cudaError_t res = cudaGetDeviceProperties(&dev_prop, get_active_device());
    if ((res != cudaSuccess) || (dev_prop.maxThreadsPerMultiProcessor == 0)) {
        LOGE(LOG_ERROR, "get_metric_throughput: Failed to get device properties: %s", cudaGetErrorString(res));
        return;
    }
    

    pthread_mutex_lock(&client_mgr_mutex);
    if (end_call_count >= 2) {
        // Calculate the total time difference between first and last call
        double start_time = calls[end_call_index].start_time.tv_sec + calls[end_call_index].start_time.tv_usec / 1e6;
	int end_index = (end_call_index + end_call_count -1 ) % MAX_CALLS;
        double end_time = calls[end_index].end_time.tv_sec + calls[end_index].end_time.tv_usec / 1e6;
        double total_time = end_time - start_time;

        // Return the call rate: number of calls per second
        *avg_function_rate =  (end_call_count - 1) / total_time;
    }

    double total_latency = 0;
    for (int i = 0; i < end_call_count; i++) {
        int index = (end_call_index + i) % MAX_CALLS;
        double start_time = calls[index].start_time.tv_sec + calls[index].start_time.tv_usec / 1e6;
        double end_time = calls[index].end_time.tv_sec + calls[index].end_time.tv_usec / 1e6;
        total_latency += (end_time - start_time);
    }

    if (end_call_count > 0) {
    	*avg_latency =  total_latency / end_call_count;
    }

    double total_resource_usage = 0.0;
    for (int i = 0; i < end_call_count; i++) {
        int index = (end_call_index + i) % MAX_CALLS;
        total_resource_usage += calls[index].resource_usage;
    }

    if (end_call_count > 0) {
    	double percentile_value =  compute_99th_percentile();
	*avg_resource_usage = (uint32_t) ceil(percentile_value / dev_prop.maxThreadsPerMultiProcessor);
    }

    //LOGE(LOG_INFO, "Metrics returned usage %d latency %d rate %d\n count %d end_count %d", *avg_resource_usage, *avg_latency, *avg_function_rate, call_count, end_call_count);
    //call_count = (int)0;
    //end_call_count = (int)0;
    //LOGE(LOG_INFO, "Metrics returned usage %d latency %d rate %d\n count %d end_count %d", *avg_resource_usage, *avg_latency, *avg_function_rate, call_count, end_call_count);
    pthread_mutex_unlock(&client_mgr_mutex);
}

static int freeResources(cricket_client* client);

int init_cpu_server_client_mgr() {
    int ret = 0;

    ret = resource_mg_init(&pid_to_xp_fd, 0);
    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to initialize pid_to_xp_fd resource manager");
        return ret;
    }

    ret = resource_mg_init(&xp_fd_to_client, 0);
    if (ret != 0) {
        resource_mg_free(&pid_to_xp_fd);
        LOGE(LOG_ERROR, "Failed to initialize xp_fd_to_client resource manager");
        return ret;
    }
    ret = resource_mg_init(&restored_clients, 0);
    if (ret != 0) {
        resource_mg_free(&pid_to_xp_fd);
        resource_mg_free(&xp_fd_to_client);
        LOGE(LOG_ERROR, "Failed to initialize restored_clients resource manager");
        return ret;
    }

    return ret;
}

void free_cpu_server_client_mgr() {
    resource_mg_free(&pid_to_xp_fd);
    resource_mg_free(&xp_fd_to_client);
    resource_mg_free(&restored_clients);
}

cricket_client* create_client(int pid) {

    cricket_client* client = (cricket_client*)malloc(sizeof(cricket_client));
    if (client == NULL) {
        LOGE(LOG_ERROR, "Failed to allocate memory for new client");
        return NULL;
    }
    client->pid = pid;

    
    client->default_stream = NULL; // create a default stream

    cudaError_t err = cudaStreamCreateWithFlags((cudaStream_t *)&client->default_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        LOGE(LOG_ERROR, "Failed to create default stream for new client: %s", cudaGetErrorString(err));
        free(client);
        return NULL;
    }

    client->custom_streams = init_resource_map(INIT_STREAM_SLOTS);
    if (client->custom_streams == NULL) {
        cudaStreamDestroy(client->default_stream);
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map for new client");
        free(client);
        return NULL;
    }

    client->gpu_events = init_resource_map(INIT_VAR_SLOTS);
    if (client->gpu_events == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map for new client");
        //free_resource_map(client->gpu_mem);
        free_resource_map(client->custom_streams);
        free(client);
        return NULL;
    }
    resource_mg_init(&client->gpu_mem, 0);
    resource_mg_init(&client->modules, 0);
    resource_mg_init(&client->functions, 0);
    resource_mg_init(&client->vars, 0);

    client->events = init_resource_map(INIT_EVENT_SLOTS);
    if (client->events == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize events resource map for new client");
        cudaStreamDestroy(client->default_stream);
        free_resource_map(client->custom_streams);
        resource_mg_free(&client->gpu_mem);
        resource_mg_free(&client->modules);
        resource_mg_free(&client->functions);
        resource_mg_free(&client->vars);
        free(client);
        return NULL;
    }

    client->malloc_idx = 0;

    LOGE(LOG_INFO, "added client for pid %d\n", pid);

    return client;

}

int add_new_client(int pid, int xp_fd) {

    cricket_client* client = create_client(pid);
    if (client == NULL) {
        LOGE(LOG_ERROR, "Failed to create new client for pid %d", pid);
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);

    int ret = resource_mg_add_sorted(&pid_to_xp_fd, (void *)(long)pid, (void *)(long)xp_fd);
    ret |= resource_mg_add_sorted(&xp_fd_to_client, (void *)(long)xp_fd, client);

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to add new client to resource managers");
        pthread_mutex_unlock(&client_mgr_mutex);
        resource_mg_free(&client->gpu_mem);
        free_resource_map(client->custom_streams);
        free_resource_map(client->gpu_events);
        resource_mg_free(&client->modules);
        resource_mg_free(&client->functions);
        resource_mg_free(&client->vars);
        free(client);
        return -1;
    }
    pthread_mutex_unlock(&client_mgr_mutex);

    return (ret != 0)? -1:0;
}

int add_restored_client(cricket_client *client) {
    pthread_mutex_lock(&client_mgr_mutex);
    int ret = resource_mg_add_sorted(&restored_clients, (void *)(long)client->pid, (void *)client);
    pthread_mutex_unlock(&client_mgr_mutex);
    return ret;
}

int move_restored_client(int pid, int xp_fd) {
    cricket_client* client = (cricket_client*)resource_mg_get_default(&restored_clients, (void *)(long)pid, NULL);
    
    if (client == NULL) {
        LOGE(LOG_ERROR, "Client with pid %d not found in restored clients", pid);
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);
    int ret = resource_mg_add_sorted(&pid_to_xp_fd, (void *)(long)client->pid, (void *)(long)xp_fd);
    ret &= resource_mg_add_sorted(&xp_fd_to_client, (void *)(long)xp_fd, client);
    pthread_mutex_unlock(&client_mgr_mutex);

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to add restored client to resource managers");
        return -1;
    }

    return resource_mg_remove(&restored_clients, (void *)(long)pid);
}


inline cricket_client* get_client(int xp_fd) {
	cricket_client *ret = (cricket_client*)resource_mg_get_default(&xp_fd_to_client, (void *)(long)xp_fd, NULL);
	if(ret == NULL) {
	    LOGE(LOG_ERROR, "get client for %d\n", xp_fd);
	    resource_mg_print(&xp_fd_to_client);
	}
	return ret;
}

cricket_client* get_client_by_pid(int pid) {
    int xp_fd = (int)(long)resource_mg_get_default(&pid_to_xp_fd, (void *)(long)pid, (void*)-1ll);
    if (xp_fd == -1) {
        return NULL;
    }
    return get_client(xp_fd);
}

int remove_client_ptr(cricket_client* client) {
    if (client == NULL) {
        LOGE(LOG_ERROR, "Client is null");
        return -1;
    }

    pthread_mutex_lock(&client_mgr_mutex);
    LOGE(LOG_INFO, "removing client ptr from %d \n", client->pid);
    resource_mg_remove(&pid_to_xp_fd, (void *)(long)client->pid);
    pthread_mutex_unlock(&client_mgr_mutex);
    
    // need to free gpu resources and custom streams
    freeResources(client);
    

    // free client
    resource_mg_free(&client->gpu_mem);
    free_resource_map(client->custom_streams);
    free_resource_map(client->gpu_events);

    //pthread_mutex_lock(&client->modules.mutex);
    //pthread_mutex_lock(&client->modules.map_res.mutex);
    size_t i = 0;
    resource_mg_map_elem *elem = NULL;
    while(resource_mg_get_element_at(&client->modules, FALSE, i, (void **)&elem) == 0) {
        i++;
	if(elem != NULL) {
                addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
                free_module_data(pair);
	}
    }
    //pthread_mutex_unlock(&client->modules.map_res.mutex);
    //pthread_mutex_unlock(&client->modules.mutex);

    //pthread_mutex_lock(&client->vars.mutex);
    //pthread_mutex_lock(&client->vars.map_res.mutex);
    i = 0;
    elem = NULL;
    while(resource_mg_get_element_at(&client->vars, FALSE, i, (void **)&elem) == 0) {
        i++;
	if(elem != NULL) {

            addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
        
            free_variable_data(pair);
	}
    }
    //pthread_mutex_unlock(&client->vars.map_res.mutex);
    //pthread_mutex_unlock(&client->vars.mutex);

    //pthread_mutex_lock(&client->functions.mutex);
    //pthread_mutex_lock(&client->functions.map_res.mutex);
    i = 0;
    elem = NULL;
    while(resource_mg_get_element_at(&client->functions, FALSE, i, (void **)&elem) == 0) {
        i++;
	if(elem != NULL) {
                addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
                free_function_data(pair);
        }
    }
    //pthread_mutex_unlock(&client->functions.map_res.mutex);
    //pthread_mutex_unlock(&client->functions.mutex);

    resource_mg_free(&client->modules);
    resource_mg_free(&client->vars);
    resource_mg_free(&client->functions);

    free(client);
    return 0;
}

int remove_client(int xp_fd) {
    cricket_client* client = get_client(xp_fd);
    if (client == NULL) {
        LOGE(LOG_ERROR, "Client with xp_fd %d not found", xp_fd);
        return -1;
    }

    LOGE(LOG_INFO, "Client with xp_fd %d being removed", xp_fd);
    int ret = remove_client_ptr(client);

    if (ret != 0) {
        LOGE(LOG_ERROR, "Failed to remove client with xp_fd %d", xp_fd);
        return -1;
    }

    return resource_mg_remove(&xp_fd_to_client, (void *)(long)xp_fd);
}

static int freeResources(cricket_client* client) {
    PRIMARY_CTX_RETAIN;
    
    
    resource_mg_map_elem *map_elem;

    for (size_t i = 0; i < client->gpu_mem.map_res.length; i++) {
        if (resource_mg_get_element_at(&client->gpu_mem, FALSE, i, (void **)&map_elem) == 0) {
            mem_alloc_args_t *mem_args = (mem_alloc_args_t *)map_elem->cuda_address;
            free_dev_mem(map_elem->client_address, mem_args->padded_size);
	}
    }

    resource_map_iter *event_itr = resource_map_init_iter(client->gpu_events);
    
    if (event_itr == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize gpu_events map iterator");
        return -1;
    }
    uint64_t event_idx;
    while ((event_idx = resource_map_iter_next(event_itr)) != 0) {
        cudaEventDestroy((cudaEvent_t )resource_map_get_addr(client->gpu_events, (void *)event_idx));
    }

    resource_map_free_iter(event_itr);

    resource_map_iter *stream_iter = resource_map_init_iter(client->custom_streams);
    
    if (stream_iter == NULL) {
        LOGE(LOG_ERROR, "Failed to initialize custom_streams resource map iterator");
        return -1;
    }

    uint64_t stream_idx;
    while ((stream_idx = resource_map_iter_next(stream_iter)) != 0) {
        cudaStreamDestroy((cudaStream_t)resource_map_get_addr(client->custom_streams, (void *)stream_idx));
    }

    resource_map_free_iter(stream_iter);

    cudaStreamDestroy(client->default_stream);
    

    PRIMARY_CTX_RELEASE;
}



cricket_client_iter get_client_iter() {
	    resource_mg_print(&xp_fd_to_client);
    return 0;
}

cricket_client* get_next_client(cricket_client_iter* iter) {
    
    if (iter == NULL) {
        LOGE(LOG_ERROR, "get_next_client iter is NULL ");
        return NULL;
    }

    resource_mg_map_elem *elem;
    if(resource_mg_get_element_at(&xp_fd_to_client, FALSE, *iter, (void **)&elem) != 0) {
            LOGE(LOG_INFO, "get_next_client resource mgt_get_element returned NULL ");
	    return NULL;
    }
    *iter += 1;
        LOGE(LOG_DEBUG, "get_next_client returned %p ", elem->cuda_address);
    return (cricket_client *)elem->cuda_address;
}


cricket_client* get_next_restored_client(cricket_client_iter* iter) {
    
    if (iter == NULL) {
        return NULL;
    }

    resource_mg_map_elem *elem;
    if(resource_mg_get_element_at(&restored_clients, FALSE, *iter, (void **)&elem) != 0) {
	    return NULL;
    }
    *iter += 1;
    return (cricket_client *)elem->cuda_address;
}

void free_variable_data(addr_data_pair_t *pair) {
    switch (pair->reg_data.type) {
        case MODULE_GET_GLOBAL:
            var_register_args_t *data = (var_register_args_t *)pair->reg_data.data;
            free(data->deviceName); // it is an string arg
            if (data->data) {
                free(data->data);
            }
            free(pair->reg_data.data);
            break;
        default:
            LOGE(LOG_ERROR, "Invalid variable data type: %d", pair->reg_data.type);
            break;
    }
    free(pair);
}

void free_function_data(addr_data_pair_t *pair) {

    switch (pair->reg_data.type) {
        case MODULE_GET_FUNCTION:
            rpc_cumodulegetfunction_1_argument *data = (rpc_cumodulegetfunction_1_argument *)pair->reg_data.data;
            free(data->arg2); // it is an string arg
            free(pair->reg_data.data);
            break;
        default:
            LOGE(LOG_ERROR, "Invalid function data type: %d", pair->reg_data.type);
            break;
    }
    free(pair);
}


void free_module_data(addr_data_pair_t *pair) {
    
    switch (pair->reg_data.type) {
        case MODULE_LOAD:
            free(pair->reg_data.data);
            break;
        case MODULE_LOAD_DATA:
            mem_data* data = (mem_data *)pair->reg_data.data;
            free(data->mem_data_val);
            free(pair->reg_data.data);
            break;
        default:
            LOGE(LOG_ERROR, "Invalid module data type: %d", pair->reg_data.type);
            break;
    }
    free(pair);
}

int fetch_variable_data_to_host(void) {

    pthread_mutex_lock(&client_mgr_mutex);
    cricket_client_iter iter = get_client_iter();

    cricket_client* client;

    while ((client = get_next_client(&iter)) != NULL) {

        //pthread_mutex_lock(&client->vars.mutex);
        //pthread_mutex_lock(&client->vars.map_res.mutex);
	size_t i = 0;
        resource_mg_map_elem *elem = NULL;
	resource_mg_print(&client->vars);
        while(resource_mg_get_element_at(&(client->vars), FALSE, i, (void **)&elem) == 0) {
            i++;
	    if(elem != NULL) {

                addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;
            
                var_register_args_t *data = (var_register_args_t *)pair->reg_data.data;

                if (data->data == NULL) {
                    data->data = malloc(data->size);
                }

                if (cudaMemcpyFromSymbol(data->data, pair->addr, data->size, 0, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    LOGE(LOG_ERROR, "Failed to copy variable data to host");
                    //pthread_mutex_unlock(&client->vars.map_res.mutex);
                    //pthread_mutex_unlock(&client->vars.mutex);
    		    pthread_mutex_unlock(&client_mgr_mutex);
                    return -1;
                }
	    }
        }
        //pthread_mutex_unlock(&client->vars.map_res.mutex);
        //pthread_mutex_unlock(&client->vars.mutex);
    }
    pthread_mutex_unlock(&client_mgr_mutex);
    return 0;
}

int dump_module_data(resource_mg_map_elem *elem, FILE* fp) {

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    fwrite(elem, sizeof(resource_mg_map_elem), 1, fp);
    fwrite(pair, sizeof(addr_data_pair_t), 1, fp);

    switch (pair->reg_data.type) {
        case MODULE_LOAD:
            {
                char* data = (char *)pair->reg_data.data;
                fwrite(data, sizeof(char), pair->reg_data.size, fp);
                break;
            }
        case MODULE_LOAD_DATA:
            {
                mem_data* data = (mem_data *)pair->reg_data.data;
                fwrite(data, sizeof(mem_data), 1, fp);
                fwrite(data->mem_data_val, data->mem_data_len, 1, fp);
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid module data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int load_module_data(resource_mg_map_elem *elem, FILE* fp) {
    size_t readsz;
    readsz = fread(elem, sizeof(resource_mg_map_elem), 1, fp);

    if (readsz != 1) {
        return -1;
    }

    elem->cuda_address = malloc(sizeof(addr_data_pair_t));
    readsz = fread(elem->cuda_address, sizeof(addr_data_pair_t), 1, fp);

    if (readsz != 1) {
        free(elem->cuda_address);
        return -1;
    }

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    switch (pair->reg_data.type) {
        case MODULE_LOAD:
            {
                char* data = (char *)malloc(pair->reg_data.size);
                readsz = fread(data, sizeof(char), pair->reg_data.size, fp);
                if (readsz != pair->reg_data.size) {
                    free(data);
                    free(pair);
                    return -1;
                }
                pair->reg_data.data = data;
                break;
            }
        case MODULE_LOAD_DATA:
            {
                mem_data* data = (mem_data *)malloc(sizeof(mem_data));
                readsz = fread(data, sizeof(mem_data), 1, fp);

                if (readsz != 1) {
                    free(data);
                    free(pair);
                    return -1;
                }

                data->mem_data_val = (char *)malloc(data->mem_data_len);
                readsz = fread(data->mem_data_val, data->mem_data_len, 1, fp);
                
                if (readsz != 1) {
                    free(data->mem_data_val);
                    free(data);
                    free(pair);
                    return -1;
                }

                pair->reg_data.data = data;
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid module data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int dump_variable_data(resource_mg_map_elem *elem, FILE* fp) {

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    fwrite(elem, sizeof(resource_mg_map_elem), 1, fp);
    fwrite(pair, sizeof(addr_data_pair_t), 1, fp);

    char str_dump[STR_DUMP_SIZE];

    switch (pair->reg_data.type) {
        case MODULE_GET_GLOBAL:
            {
                var_register_args_t *data = (var_register_args_t *)pair->reg_data.data;
                fwrite(data, sizeof(var_register_args_t), 1, fp);
                strncpy(str_dump, data->deviceName, STR_DUMP_SIZE);
                fwrite(str_dump, sizeof(char), STR_DUMP_SIZE, fp);

                size_t var_size = data->size;
                char *var_data = (char *)malloc(var_size);

                cudaMemcpyFromSymbol(var_data, pair->addr, var_size, 0, cudaMemcpyDeviceToHost);

                fwrite(var_data, var_size, 1, fp);

                free(var_data);

                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid variable data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int load_variable_data(resource_mg_map_elem *elem, FILE *fp) {
    size_t readsz;
    readsz = fread(elem, sizeof(resource_mg_map_elem), 1, fp);

    if (readsz != 1) {
        return -1;
    }

    elem->cuda_address = malloc(sizeof(addr_data_pair_t));
    readsz = fread(elem->cuda_address, sizeof(addr_data_pair_t), 1, fp);

    if (readsz != 1) {
        free(elem->cuda_address);
        return -1;
    }

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    switch (pair->reg_data.type) {
        case MODULE_GET_GLOBAL:
            {
                var_register_args_t *data = (var_register_args_t *)malloc(sizeof(var_register_args_t));
                readsz = fread(data, sizeof(var_register_args_t), 1, fp);

                if (readsz != 1) {
                    free(data);
                    free(pair);
                    return -1;
                }

                data->deviceName = (char *)malloc(STR_DUMP_SIZE);
                readsz = fread(data->deviceName, sizeof(char), STR_DUMP_SIZE, fp);
                
                if (readsz != STR_DUMP_SIZE) {
                    free(data->deviceName);
                    free(data);
                    free(pair);
                    return -1;
                }

                data->data = malloc(data->size);
                readsz = fread(data->data, data->size, 1, fp);

                if (readsz != 1) {
                    free(data->data);
                    free(data->deviceName);
                    free(data);
                    free(pair);
                    return -1;
                }

                

                pair->reg_data.data = data;
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid variable data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int dump_function_data(resource_mg_map_elem *elem, FILE* fp) {

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    fwrite(elem, sizeof(resource_mg_map_elem), 1, fp);
    fwrite(pair, sizeof(addr_data_pair_t), 1, fp);

    char str_dump[STR_DUMP_SIZE];

    switch (pair->reg_data.type) {
        case MODULE_GET_FUNCTION:
            {
                rpc_cumodulegetfunction_1_argument *data = (rpc_cumodulegetfunction_1_argument *)pair->reg_data.data;
                fwrite(data, sizeof(rpc_cumodulegetfunction_1_argument), 1, fp);
                strncpy(str_dump, data->arg2, STR_DUMP_SIZE);
                fwrite(str_dump, sizeof(char), STR_DUMP_SIZE, fp);
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid function data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}

int load_function_data(resource_mg_map_elem *elem, FILE *fp) {
    size_t readsz;
    readsz = fread(elem, sizeof(resource_mg_map_elem), 1, fp);

    if (readsz != 1) {
        return -1;
    }

    elem->cuda_address = malloc(sizeof(addr_data_pair_t));
    readsz = fread(elem->cuda_address, sizeof(addr_data_pair_t), 1, fp);

    if (readsz != 1) {
        free(elem->cuda_address);
        return -1;
    }

    addr_data_pair_t *pair = (addr_data_pair_t *)elem->cuda_address;

    switch (pair->reg_data.type) {
        case MODULE_GET_FUNCTION:
            {
                rpc_cumodulegetfunction_1_argument *data = (rpc_cumodulegetfunction_1_argument *)malloc(sizeof(rpc_cumodulegetfunction_1_argument));
                readsz = fread(data, sizeof(rpc_cumodulegetfunction_1_argument), 1, fp);

                if (readsz != 1) {
                    free(data);
                    free(pair);
                    return -1;
                }

                data->arg2 = (char *)malloc(STR_DUMP_SIZE);
                readsz = fread(data->arg2, sizeof(char), STR_DUMP_SIZE, fp);
                
                if (readsz != STR_DUMP_SIZE) {
                    free(data->arg2);
                    free(data);
                    free(pair);
                    return -1;
                }

                pair->reg_data.data = data;
                break;
            }
        default:
            LOGE(LOG_ERROR, "Invalid function data type: %d", pair->reg_data.type);
            return -1;
    }
    return 0;
}


int dealloc_client_resources() {
    cricket_client_iter iter = get_client_iter();

    cricket_client* client;

    while ((client = get_next_client(&iter)) != NULL) {
        freeResources(client);
    }

    return 0;
}
