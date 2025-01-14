#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <pthread.h>

#include "log.h"
#include "cpu-server-metrics.h"
#include "device-management.h"
#include "msg-handler.h"

extern pthread_mutex_t metrics_mutex;

// SM Core metrics`
static struct metricsData kernel_resource_usages[MAX_METRICS_ENTRIES];
static int metrics_kernel_count = 0, metrics_kernel_start_index = 0;
static int streams_count = 0, events_count = 0;

uint64_t kernel_total_resource = 0;  // Total resource utilization
uint64_t kernel_first_start_time = 0;  // Start time of the oldest entry
uint64_t kernel_last_end_time = 0;     // End time of the newest entry

// Memcopy metrics
static struct metricsData memcpy_resource_usages[MAX_METRICS_ENTRIES];
static int metrics_memcpy_count = 0, metrics_memcpy_start_index = 0;

uint64_t memcpy_total_resource = 0;  // Total resource utilization
uint64_t memcpy_first_start_time = 0;  // Start time of the oldest entry
uint64_t memcpy_last_end_time = 0;     // End time of the newest entry


// Context count and StreamCount

// Get current time in microseconds
uint64_t get_current_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

void get_client_metrics(struct metricsInfo *info) {
    pthread_mutex_lock(&metrics_mutex);

    info->totalThreadResource = kernel_total_resource;
    info->kernelElapseTimeMS = (double)(kernel_last_end_time - kernel_first_start_time) / 1000.0;
    info->kernelLaunchCount = metrics_kernel_count;

    info->memUsage = get_gpu_memory_usage();

    info->totalMemcpySize = memcpy_total_resource;
    info->memcpyElapseTimeMS = (double)(memcpy_last_end_time - memcpy_first_start_time) / 1000.0;
    info->memcpyCount = metrics_memcpy_count;

    info->totalStreams = streams_count;
    info->totalEvents = events_count;

    pthread_mutex_unlock(&metrics_mutex);
}

void record_resource_acquisition(cricket_client *client, int resources) {

    pthread_mutex_lock(&metrics_mutex);

    int write_index = (metrics_kernel_start_index + metrics_kernel_count) % MAX_METRICS_ENTRIES;  // Determine the next write position

    // Handle cumulative totals and timestamps
    if (metrics_kernel_count == MAX_METRICS_ENTRIES) {
        // Subtract the resource of the oldest entry from the total
        kernel_total_resource -= kernel_resource_usages[metrics_kernel_start_index].resource;
        // Move the start index to the next oldest entry
        metrics_kernel_start_index = (metrics_kernel_start_index + 1) % MAX_METRICS_ENTRIES;
    } else {
        metrics_kernel_count++;
    }

    // Add the new resource usage to the buffer
    kernel_resource_usages[write_index].start_time = get_current_time();
    kernel_resource_usages[write_index].resource = resources;

    // Update cumulative totals and timestamps
    kernel_total_resource += resources;
    kernel_first_start_time = memcpy_resource_usages[metrics_kernel_start_index].start_time;  // First entry
    kernel_last_end_time = memcpy_resource_usages[write_index].start_time;  // Latest entry

    // Update streams count and events count
    streams_count = client->custom_streams->length;
    events_count = client->gpu_events->length;
    LOGE(LOG_INFO, "resource acquired: threads %ld, launchcount %ld, stream_count %ld, events_count %ld, elapsetime-ms %ld",
		    kernel_total_resource, metrics_kernel_count, streams_count, events_count,
    		    (kernel_last_end_time - kernel_first_start_time) / 1000.0);
    pthread_mutex_unlock(&metrics_mutex);
}

void record_memcpy_resource_acquisition(cricket_client *client, int copysize) {

    pthread_mutex_lock(&metrics_mutex);

    int write_index = (metrics_memcpy_start_index + metrics_memcpy_count) % MAX_METRICS_ENTRIES;  // Determine the next write position

    // Handle cumulative totals and timestamps
    if (metrics_memcpy_count == MAX_METRICS_ENTRIES) {
        // Subtract the resource of the oldest entry from the total
        memcpy_total_resource -= memcpy_resource_usages[metrics_memcpy_start_index].resource;
        // Move the start index to the next oldest entry
        metrics_memcpy_start_index = (metrics_memcpy_start_index + 1) % MAX_METRICS_ENTRIES;
    } else {
        metrics_memcpy_count++;
    }

    // Add the new resource usage to the buffer
    memcpy_resource_usages[write_index].start_time = get_current_time();
    memcpy_resource_usages[write_index].resource = copysize;

    // Update cumulative totals and timestamps
    memcpy_total_resource += copysize;
    memcpy_first_start_time = memcpy_resource_usages[metrics_memcpy_start_index].start_time;  // First entry
    memcpy_last_end_time = memcpy_resource_usages[write_index].start_time;  // Latest entry

    pthread_mutex_unlock(&metrics_mutex);
}
