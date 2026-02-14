/* Copyright (c) 2024-2026 SynerG Lab, IITB */

#ifndef _CPU_SERVER_METRICS_H_
#define _CPU_SERVER_METRICS_H_

// Metric measurement throughput
#include <stdint.h>
#include "cpu-server-client-mgr.h"
#include "msg-handler.h"

// #include <time.h>
#define MAX_METRICS_ENTRIES 100
#define MAX_METRICS_INTERVAL 100 // mill seconds

struct metricsData {
    uint64_t start_time;
    uint64_t resource; // threads in this case.
};


uint64_t get_current_time();
void get_client_metrics(struct metricsInfo *info);
void record_resource_acquisition(cricket_client *client, int resources);
void record_memcpy_resource_acquisition(cricket_client *client, int copysize);

#endif // _CPU_SERVER_METRICS_H_
