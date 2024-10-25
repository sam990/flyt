// Copyright 2021-2022 NVIDIA Corporation. All rights reserved
//
// This sample demostrates using the profiler API in injection mode.
// Build this file as a shared object, and set environment variable
// CUDA_INJECTION64_PATH to the full path to the .so.
//
// CUDA will load the object during initialization and will run
// the function called 'InitializeInjection'.
//
// After the initialization routine  returns, the application resumes running,
// with the registered callbacks triggering as expected.  These callbacks
// are used to start a Profiler API session using Kernel Replay and
// Auto Range modes.
//
// A configurable number of kernel launches (default 10) are run
// under one session.  Before the 11th kernel launch, the callback
// ends the session, prints metrics, and starts a new session.
//
// An atexit callback is also used to ensure that any partial sessions
// are handled when the target application exits.
//
// This code supports multiple contexts and multithreading through
// locking shared data structures.

// System headers
#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <mutex>
using ::std::mutex;

#include <string>
using ::std::string;

#include <unordered_map>
using ::std::unordered_map;

#include <unordered_set>
using ::std::unordered_set;

#include <vector>
using ::std::vector;

#include <stdlib.h>
#include <unordered_map>
#include <deque>



#include "profiler.h"
#include "helper_cupti.h"

// NVPW headers
#include <nvperf_host.h>

#include <Eval.h>
using ::NV::Metric::Eval::PrintMetricValues;
using ::NV::Metric::Eval::GetMetricGpuValue;

#include <Metric.h>
using ::NV::Metric::Config::GetConfigImage;
using ::NV::Metric::Config::GetCounterDataPrefixImage;

#include <Utils.h>
using ::NV::Metric::Utils::GetNVPWResultString;


// Macros
// Export InitializeInjection symbol.
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#define HIDDEN
#else
#define DLLEXPORT __attribute__((visibility("default")))
#define HIDDEN __attribute__((visibility("hidden")))
#endif

struct FunctionMetrics {
    uint64_t totalThreads;
    uint64_t sharedMem;
    uint64_t regSize;
	std::deque<std::pair<uint64_t, uint64_t>> timestamps; // FIFO list for start and end times

	FunctionMetrics() : totalThreads(0), sharedMem(0), regSize(0), timestamps() {}

};

/*
// Define a structure to hold the metric data
struct MetricData {
    double meanValue; // Stores the harmonic mean computed so far
    uint64_t count;      // Number of values processed so far
	bool isHarmonic;

	 // Constructor to initialize MetricData
    MetricData() : meanValue(0.0), count(0), isHarmonic(false) {}

    // Function to compute harmonic mean incrementally
    void updateMetricsMean(uint64_t newValue, uint64_t interval) {
			if(isHarmonic) {
					updateHarmonicMean(newValue, interval);
			}
			else {
					double weightedMean = interval * meanValue;
					count += interval;

					meanValue = weightedMean/count;
			}
	}
	void setIsHarmonic(bool isHarmonicValue) {
			isHarmonic = isHarmonicValue;
	}
    private void updateHarmonicMean(uint64_t newValue, uint64_t interval) {
        if (count == 0) {
            // First value, initialize the harmonic mean
            meanValue = static_cast<double>(newValue);
			count = interval
        } else {
            // Update the harmonic mean incrementally
            double newHarmonicMean = (count * (1.0 / meanValue)) + (1.0 / static_cast<double>(newValue));
			count += interval;
            meanValue = (count) / newHarmonicMean;
        }

    }

	uint64_t getMetricMean() {
			return (uint64_t)meanValue;
	}
};
*/


// Profiler API configuration data, per-context.
struct CtxProfilerData
{
    CUcontext       ctx;
    int             deviceId;
    cudaDeviceProp  deviceProp;
    vector<uint8_t> counterAvailabilityImage;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    vector<uint8_t> counterDataImage;
    vector<uint8_t> counterDataPrefixImage;
    vector<uint8_t> counterDataScratchBufferImage;
    vector<uint8_t> configImage;
    int             maxNumRanges;
    int             curRanges;
    int             maxRangeNameLength;
    // Count of sessions.
    int             iterations;
    // Information of functions.
    std::unordered_map<std::string, FunctionMetrics> functionInfo;
    std::unordered_map<std::string, std::deque<uint64_t>> metricsMap;
	std::deque<std::pair<uint64_t, uint64_t>> timestamps; // FIFO list for start and end times

    // Initialize fields, with env var overrides.
    CtxProfilerData() : curRanges(), maxRangeNameLength(64), iterations(), functionInfo(), metricsMap(), timestamps()
    {
        char *pEnvVar = getenv("INJECTION_KERNEL_COUNT");
        if (pEnvVar != NULL)
        {
            int value = atoi(pEnvVar);
            if (value < 1)
            {
                cerr << "Read " << value << " kernels from INJECTION_KERNEL_COUNT, but must be >= 1; defaulting to 10." << endl;
                value = 10;
            }
            maxNumRanges = value;
        }
        else
        {
            maxNumRanges = 10;
        }
    };

    // Add a new record
	void addFunctionRecord(const std::string& functionName, uint64_t threads, uint64_t mem, uint64_t regs) {
		FunctionMetrics &info = functionInfo[functionName];
		info.totalThreads = threads;
		info.sharedMem = mem;
		info.regSize = regs;
		uint64_t starttime;
        CUPTI_API_CALL(cuptiGetTimestamp(&starttime));
		// Add start time and a placeholder for end time
        info.timestamps.push_back({starttime, 0});
	}

	// Update the earliest record's end time for a given function
	void updateFunctionEndTime(const std::string& functionName) {
		auto& info = functionInfo[functionName];
		uint64_t endtime;
        CUPTI_API_CALL(cuptiGetTimestamp(&endtime));

		for (auto &timestamp : info.timestamps) {
			cout << " function name " << functionName << " startime " << timestamp.first << " end time " << timestamp.second << " end time " << endtime << endl;
            if (timestamp.second == 0) { // Check for uninitialized end time
                timestamp.second = endtime;
                return; // Exit after the first uninitialized end time is set
            }
        }
	}

	// Function to add or update a metric
	void addOrUpdateMetric(const std::string& metricsName, uint64_t value) {
		auto& array = metricsMap[metricsName];
		array.push_back(value);
	}


};

// Track per-context profiler API data in a shared map.
static mutex ctxDataMutex;
static unordered_map<CUcontext, CtxProfilerData> contextData;
static bool injectionInitialized = false;

// List of metrics to collect.
static vector<string> metricNames;

// Initialize state.
void
InitializeState()
{
    static int profilerInitialized = 0;

    if (profilerInitialized == 0)
    {
        // CUPTI Profiler API initialization
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        // NVPW required initialization
        NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
        NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

        profilerInitialized = 1;
		cout << endl << "Initilized profiler ############ " << endl;
    }
}

// Initialize profiler for a context.
void
InitializeContextData(
    CtxProfilerData &contextData)
{
    InitializeState();

    // Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data.
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Allocate sized counterAvailabilityImage.
    contextData.counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);

    // Initialize counterAvailabilityImage.
    getCounterAvailabilityParams.pCounterAvailabilityImage = contextData.counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Fill in configImage - can be run on host or target.
    if (!GetConfigImage(contextData.deviceProp.name, metricNames, contextData.configImage, contextData.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create configImage for context " << contextData.ctx << endl;
        exit(EXIT_FAILURE);
    }

    // Fill in counterDataPrefixImage - can be run on host or target.
    if (!GetCounterDataPrefixImage(contextData.deviceProp.name, metricNames, contextData.counterDataPrefixImage, contextData.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create counterDataPrefixImage for context " << contextData.ctx << endl;
        exit(EXIT_FAILURE);
    }

    // Record counterDataPrefixImage info and other options for sizing the counterDataImage.
    contextData.counterDataImageOptions.pCounterDataPrefix = contextData.counterDataPrefixImage.data();
    contextData.counterDataImageOptions.counterDataPrefixSize = contextData.counterDataPrefixImage.size();
    contextData.counterDataImageOptions.maxNumRanges = contextData.maxNumRanges;
    contextData.counterDataImageOptions.maxNumRangeTreeNodes = contextData.maxNumRanges;
    contextData.counterDataImageOptions.maxRangeNameLength = contextData.maxRangeNameLength;

    // Calculate size of counterDataImage based on counterDataPrefixImage and options.
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &(contextData.counterDataImageOptions);
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
    // Create counterDataImage.
    contextData.counterDataImage.resize(calculateSizeParams.counterDataImageSize);

    // Initialize counterDataImage inside StartSession.
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(contextData.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = contextData.counterDataImage.size();
    initializeParams.pCounterDataImage = contextData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    // Calculate scratchBuffer size based on counterDataImage size and counterDataImage.
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = contextData.counterDataImage.size();
    scratchBufferSizeParams.pCounterDataImage = contextData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    // Create counterDataScratchBuffer
    contextData.counterDataScratchBufferImage.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    // Initialize counterDataScratchBuffer.
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = contextData.counterDataImage.size();
    initScratchBufferParams.pCounterDataImage = contextData.counterDataImage.data();
    initScratchBufferParams.counterDataScratchBufferSize = contextData.counterDataScratchBufferImage.size();;
    initScratchBufferParams.pCounterDataScratchBuffer = contextData.counterDataScratchBufferImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

	cout << endl << "Initilized Ontext Data ############ " << endl;

}

// Start a session.
void
StartSession(
    CtxProfilerData &contextData)
{
    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
    beginSessionParams.counterDataImageSize = contextData.counterDataImage.size();
    beginSessionParams.pCounterDataImage = contextData.counterDataImage.data();
    beginSessionParams.counterDataScratchBufferSize = contextData.counterDataScratchBufferImage.size();
    beginSessionParams.pCounterDataScratchBuffer = contextData.counterDataScratchBufferImage.data();
    beginSessionParams.ctx = contextData.ctx;
    beginSessionParams.maxLaunchesPerPass = contextData.maxNumRanges;
    beginSessionParams.maxRangesPerPass = contextData.maxNumRanges;
    beginSessionParams.pPriv = NULL;
    beginSessionParams.range = CUPTI_AutoRange;
    beginSessionParams.replayMode = CUPTI_KernelReplay;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = contextData.configImage.data();
    setConfigParams.configSize = contextData.configImage.size();
    // Only set for Application Replay mode
    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = 1;
    setConfigParams.targetNestingLevel = 1;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    enableProfilingParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));

	/* Add timestamp */
	uint64_t starttime;
    CUPTI_API_CALL(cuptiGetTimestamp(&starttime));
    contextData.timestamps.push_back({starttime, 0});
	for (const auto& metric : metricNames) {
		contextData.addOrUpdateMetric(metric, true);
    }

    contextData.iterations++;
}

// Print session data
static void
UpdateData(
    CtxProfilerData &contextData)
{
	std::vector<NV::Metric::Eval::MetricNameValue> metricNameValueMap;
	uint64_t endtime;
    cout << endl << "Context " << contextData.ctx << ", device " << contextData.deviceId << " (" << contextData.deviceProp.name << ") session " << contextData.iterations << ":" << endl;
    GetMetricGpuValue(contextData.deviceProp.name, contextData.counterDataImage, metricNames, metricNameValueMap, contextData.counterAvailabilityImage.data());
	for (const auto& metric : metricNameValueMap) {
		double value = 0;
		size_t count = 0;
		for (const auto &metricValue: metric.rangeNameMetricValueMap) 
		{
			count++;
			value += metricValue.second;
		}
		cout << endl << "metricName " << metric.metricName << " value " << value << endl;
        contextData.addOrUpdateMetric(metric.metricName, value);
    }

    CUPTI_API_CALL(cuptiGetTimestamp(&endtime));
	/* Update kernel end as we are stopping profiling.. */
	for (const auto& funcs : contextData.functionInfo) {
		contextData.updateFunctionEndTime(funcs.first);
    }
	for (auto &timestamp : contextData.timestamps) {
        if (timestamp.second == 0) { // Check for uninitialized end time
            timestamp.second = endtime;
            return; // Exit after the first uninitialized end time is set
        }
    }
}

static void
PrintData(
    CtxProfilerData &contextData)
{
}

// End a session during execution
void
EndSession(
    CtxProfilerData &contextData)
{
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    disableProfilingParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    unsetConfigParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
    endSessionParams.ctx = contextData.ctx;
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

    UpdateData(contextData);

    // Clear counterDataImage (otherwise it maintains previous records when it is reused)
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(contextData.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = contextData.counterDataImage.size();
    initializeParams.pCounterDataImage = contextData.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));
}

// C interface..
// order of kernel metrics = average kernel_launch throughput, average kernel_exeuction_time, average total_threads, average shareMem, average RegSize
// % metrics = wrap occupancy %, sm_efficiency %, sm_througput
// Total of 8 metrics..
int getMetrics(uint64_t *array, int size)
{
	int count = size > 8? 8: size;
	uint64_t kl_throughput = 0, ke_time = 0, kt_threads = 0, ks_mem = 0, kr_size = 0; 
	double occupancy = 0, efficiency = 0, throughput = 0;
	uint64_t total_global_count = 0;
	uint64_t total_kernel_count = 0;
	uint64_t total_sessiontime = 0;
	int result_count = 0;

    if (injectionInitialized == false)
	    return 0;
    //CUPTI_API_CALL(cuptiGetLastError());
    ctxDataMutex.lock();

    for (auto itr = contextData.begin(); itr != contextData.end(); ++itr)
    {
        CtxProfilerData &data = itr->second;

        if (data.curRanges > 0)
        {
			for (const auto& pair : data.functionInfo) {
        		const std::string& functionName = pair.first;
        		const FunctionMetrics& fnMetrics = pair.second;

       			uint64_t totalDifference = 0;
       			size_t count = 0;

       			// Iterate through the deque and compute differences
       			for (const auto& timestampPair : fnMetrics.timestamps) {
           			if (timestampPair.first < timestampPair.second) {
               			totalDifference += (timestampPair.second - timestampPair.first);
               			++count;
           			}
       			}

				kl_throughput += count;
				ke_time += totalDifference;
				kt_threads += fnMetrics.totalThreads;
				ks_mem += fnMetrics.sharedMem;
				kr_size += fnMetrics.regSize;
				total_kernel_count += count;
       		}
       		uint64_t totalDifference = 0;
			for (const auto& pair : data.metricsMap) {
				const std::string& metricName = pair.first;
                const std::deque<uint64_t> metrics = pair.second;

       			uint64_t totalInterval = 0;
       			double value = 0;
       			size_t count = 0;
				double *ptr;
				bool weight = false;

				if (metricName.find("sm__warps_active") != std::string::npos) {
					ptr = &occupancy;
					weight = true;
				}
				else if (metricName.find("smsp__cycles_active") != std::string::npos) {
					ptr = &efficiency;
					weight = false;
				}
				else if (metricName.find("sm__throughput") != std::string::npos) {
					ptr = &throughput;
					weight = true;
				}

				if (metrics.size() == data.timestamps.size()) {
					for (size_t metricIndex = 0; metricIndex < metrics.size(); ++metricIndex)
					{
						uint64_t metricValue = metrics[metricIndex];
						uint64_t interval = 0;
						auto &timePair = data.timestamps[metricIndex];

						if(timePair.first <= timePair.second) {
								interval = timePair.second - timePair.first;
						}

						if(interval && weight) {
							value += metricValue * interval;
							count++;
							totalInterval += interval;
						}
						else if (interval && metricValue != 0) {
							count++;
							value += 1.0/metricValue;
							totalInterval += interval;
						}
					}

					*ptr += value;
					total_sessiontime += totalInterval;
					total_global_count += count;
				}
   			}
	    }
	}
    ctxDataMutex.unlock();

	if (ke_time != 0) {
		kl_throughput /= ke_time;
		result_count++;
	}
	if(total_kernel_count != 0) {
		ke_time /= total_kernel_count;
		result_count++;
	}
	if (total_kernel_count != 0) {
		kt_threads /= total_kernel_count;
		result_count++;
	}
	if (total_kernel_count != 0) {
		ks_mem /= total_kernel_count;
		result_count++;
	}
	if (total_kernel_count != 0) {
		kr_size /= total_kernel_count;
		result_count++;
	}

	if(total_sessiontime != 0) {
		occupancy = occupancy / total_sessiontime;
		result_count++;
	    throughput = throughput / total_sessiontime;
		result_count++;
	}
	if(efficiency != 0) {
		efficiency = total_global_count/efficiency;
		result_count++;
	}

	switch (result_count) {
			case 8:
					array[7] = throughput;
			case 7:
					array[6] = efficiency;
			case 6:
					array[5] = occupancy;
			case 5:
					array[4] = kr_size;
			case 4:
					array[3] = ks_mem;
			case 3:
					array[2] = kt_threads;
			case 2:
					array[1] = ke_time;
			case 1:
					array[0] = kl_throughput;
					break;
			default:
					return 0;
	}
	return result_count;


}

// Clean up at end of execution
static void
EndExecution()
{
    CUPTI_API_CALL(cuptiGetLastError());
    ctxDataMutex.lock();

    for (auto itr = contextData.begin(); itr != contextData.end(); ++itr)
    {
        CtxProfilerData &data = itr->second;

        if (data.curRanges > 0)
        {
            PrintData(data);
            data.curRanges = 0;
        }
    }

    ctxDataMutex.unlock();
}

// Callback handler
void
ProfilerCallbackHandler(
    void *pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void const *pCallbackData)
{
    static int initialized = 0;

	cout << endl << " Profiler callback  $$$$$$$$$$$$$$$$$$$ " << endl;
    CUptiResult res;
    if (injectionInitialized == false)
	    return;

    if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
    {
        // For a driver call to launch a kernel:
        if (callbackId == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
        {
            CUpti_CallbackData const *pData = static_cast<CUpti_CallbackData const *>(pCallbackData);
            CUcontext ctx = pData->context;

            // On entry, enable / update profiling as needed
            if (pData->callbackSite == CUPTI_API_ENTER)
            {
                // Check for this context in the configured contexts
                // If not configured, it isn't compatible with profiling
                ctxDataMutex.lock();
                if (contextData.count(ctx) > 0)
                {
                    // If at maximum number of ranges, end session and reset
                    if (contextData[ctx].curRanges == contextData[ctx].maxNumRanges)
                    {
                        EndSession(contextData[ctx]);
                        contextData[ctx].curRanges = 0;
                    }

                    // If no currently enabled session on this context, start one
                    if (contextData[ctx].curRanges == 0)
                    {
                        InitializeContextData(contextData[ctx]);
                        StartSession(contextData[ctx]);
                    }

                    // Increment curRanges
                    contextData[ctx].curRanges++;
					// Update kernel information
        			CUfunction kernelFunction = (CUfunction)pData->functionName;
        			// Retrieve kernel properties
        			cudaFuncAttributes attr;
        			cudaFuncGetAttributes(&attr, kernelFunction);

					contextData[ctx].addFunctionRecord(pData->symbolName, attr.maxThreadsPerBlock, attr.sharedSizeBytes, attr.numRegs);
                }
                ctxDataMutex.unlock();
            }

            if (pData->callbackSite == CUPTI_API_EXIT)
			{
			   // Rare race condition
               if (contextData[ctx].curRanges != 0)
				   contextData[ctx].updateFunctionEndTime(pData->symbolName);
			}
        }
    }
    else if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        // When a context is created, check to see whether the device is compatible with the Profiler API
        if (callbackId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
        {
            CUpti_ResourceData const *pResourceData = static_cast<CUpti_ResourceData const *>(pCallbackData);
            CUcontext ctx = pResourceData->context;

            // Configure handler for new context under lock
            CtxProfilerData data = { };

            data.ctx = ctx;

            RUNTIME_API_CALL(cudaGetDevice(&(data.deviceId)));

            RUNTIME_API_CALL(cudaGetDeviceProperties(&(data.deviceProp), data.deviceId));

            // Initialize profiler API and test device compatibility
            InitializeState();
            CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
            params.cuDevice = data.deviceId;
            params.api = CUPTI_PROFILER_RANGE_PROFILING;
            CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

            // If valid for profiling, set up profiler and save to shared structure
            ctxDataMutex.lock();
            if (params.isSupported == CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
            {
                // Update shared structures
                contextData[ctx] = data;
                InitializeContextData(contextData[ctx]);
            }
            else
            {
                if (contextData.count(ctx))
                {
                    // Update shared structures
                    contextData.erase(ctx);
                }

                cerr << "libinjection: Unable to profile context on device " << data.deviceId << endl;

                if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice architecture is not supported" << endl;
                }

                if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice sli configuration is not supported" << endl;
                }

                if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice vgpu configuration is not supported" << endl;
                }
                else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
                {
                    cerr << "\tdevice vgpu configuration disabled profiling support" << endl;
                }

                if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    cerr << "\tdevice confidential compute configuration is not supported" << endl;
                }

                if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
                }

                if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
                {
                    ::std::cerr << "\tWSL is not supported" << ::std::endl;
                }
            }
            ctxDataMutex.unlock();
        }
    }

    return;
}

// Register callbacks for several points in target application execution
void
RegisterCallbacks( )
{
    // One subscriber is used to register multiple callback domains
    CUpti_SubscriberHandle subscriber;
    cout << "Register callback called" << endl;
    CUPTI_API_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)ProfilerCallbackHandler, NULL));
    cout << "Subscribe  callback called" << endl;
    // Runtime callback domain is needed for kernel launch callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    cout << "SEnable cuLaunchKernel  callback called" << endl;
    // Resource callback domain is needed for context creation callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));

	// Callback will be invoked at the entry and exit points of each of the CUDA Runtime API.
	//CUPTI_API_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

    cout << "SEnable Context Created  callback called" << endl;

    // Register callback for application exit
    //atexit(EndExecution);
}


// InitializeInjection will be called by the driver when the tool is loaded
// by CUDA_INJECTION64_PATH
//extern "C" DLLEXPORT int
int InitializeInjection()
{
    cout << "Initialize Metrics called" << endl;
    return 1;
    if (injectionInitialized == false)
    {

        // Read in optional list of metrics to gather
        char *pMetricEnv = getenv("INJECTION_METRICS");
        if (pMetricEnv != NULL)
        {
            char * tok = strtok(pMetricEnv, " ;,");
            do
            {
                cout << "Requesting metric '" << tok << "'" << endl;
                metricNames.push_back(string(tok));
                tok = strtok(NULL, " ;,");
            } while (tok != NULL);
        }
        else
        {
            //metricNames.push_back("sm__cycles_elapsed.avg");
            //metricNames.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.avg");
            //metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.avg");
            metricNames.push_back("sm__warps_active.avg.pct_of_peak_sustained_active");
            metricNames.push_back("smsp__cycles_active.avg.pct_of_peak_sustained_elapsed");
            metricNames.push_back("sm__throughput.avg.pct_of_peak_sustained_active");
        }

        // Subscribe to some callbacks
        RegisterCallbacks();
        injectionInitialized = true;
    }
    return 1;
}
