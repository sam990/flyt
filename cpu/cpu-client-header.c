#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

/*
// Define CUDA error codes as per the provided list
#define cudaSuccess                            0   // No errors
#define cudaErrorInvalidValue                  1   // Invalid value
#define cudaErrorMemoryAllocation              2   // Out of memory
#define cudaErrorInitializationError           3   // Driver not initialized
#define cudaErrorCudartUnloading               4   // Profiler disabled
#define cudaErrorProfilerDisabled              5   // Profiler not initialized
#define cudaErrorProfilerNotInitialized       6   // Deprecated
#define cudaErrorProfilerAlreadyStarted       7   // Profiler already started
#define cudaErrorProfilerAlreadyStopped       8   // Profiler already stopped
#define cudaErrorInvalidConfiguration         9   // Invalid configuration
#define cudaErrorInvalidPitchValue            12  // Invalid pitch value
#define cudaErrorInvalidSymbol                13  // Invalid symbol
#define cudaErrorInvalidHostPointer           16  // Invalid host pointer
#define cudaErrorInvalidDevicePointer         17  // Invalid device pointer
#define cudaErrorInvalidTexture               18  // Invalid texture
#define cudaErrorInvalidTextureBinding        19  // Invalid texture binding
#define cudaErrorInvalidChannelDescriptor     20  // Invalid channel descriptor
#define cudaErrorInvalidMemcpyDirection       21  // Invalid memcpy direction
#define cudaErrorAddressOfConstant            22  // Address of constant
#define cudaErrorTextureFetchFailed           23  // Texture fetch failed
#define cudaErrorTextureNotBound              24  // Texture not bound
#define cudaErrorSynchronizationError         25  // Synchronization error
#define cudaErrorInvalidFilterSetting         26  // Invalid filter setting
#define cudaErrorInvalidNormSetting           27  // Invalid norm setting
#define cudaErrorMixedDeviceExecution         28  // Mixed device execution
#define cudaErrorNotYetImplemented            31  // Not yet implemented
#define cudaErrorMemoryValueTooLarge          32  // Memory value too large
#define cudaErrorStubLibrary                  34  // Stub library
#define cudaErrorInsufficientDriver           35  // Insufficient driver
#define cudaErrorCallRequiresNewerDriver     36  // Requires newer driver
#define cudaErrorInvalidSurface               37  // Invalid surface
#define cudaErrorDuplicateVariableName        43  // Duplicate variable name
#define cudaErrorDuplicateTextureName         44  // Duplicate texture name
#define cudaErrorDuplicateSurfaceName         45  // Duplicate surface name
#define cudaErrorDevicesUnavailable           46  // Devices unavailable
#define cudaErrorIncompatibleDriverContext    49  // Incompatible driver context
#define cudaErrorMissingConfiguration         52  // Missing configuration
#define cudaErrorPriorLaunchFailure           53  // Prior launch failure
#define cudaErrorLaunchMaxDepthExceeded       65  // Launch max depth exceeded
#define cudaErrorLaunchFileScopedTex          66  // Launch file scoped tex
#define cudaErrorLaunchFileScopedSurf         67  // Launch file scoped surf
#define cudaErrorSyncDepthExceeded            68  // Sync depth exceeded
#define cudaErrorLaunchPendingCountExceeded   69  // Launch pending count exceeded
#define cudaErrorInvalidDeviceFunction        98  // Invalid device function
#define cudaErrorNoDevice                     100 // No device
#define cudaErrorInvalidDevice                101 // Invalid device
#define cudaErrorDeviceNotLicensed            102 // Device not licensed
#define cudaErrorSoftwareValidityNotEstablished 103 // Software validity not established
#define cudaErrorStartupFailure               127 // Startup failure
#define cudaErrorInvalidKernelImage           200 // Invalid kernel image
#define cudaErrorDeviceUninitialized          201 // Device uninitialized
#define cudaErrorContextAlreadyCurrent        202 // Context already current
#define cudaErrorMapBufferObjectFailed        205 // Map buffer object failed
#define cudaErrorUnmapBufferObjectFailed      206 // Unmap buffer object failed
#define cudaErrorArrayIsMapped                207 // Array is mapped
#define cudaErrorAlreadyMapped                208 // Already mapped
#define cudaErrorNoKernelImageForDevice       209 // No kernel image for device
#define cudaErrorAlreadyAcquired              210 // Already acquired
#define cudaErrorNotMapped                    211 // Not mapped
#define cudaErrorNotMappedAsArray             212 // Not mapped as array
#define cudaErrorNotMappedAsPointer           213 // Not mapped as pointer
#define cudaErrorECCUncorrectable             214 // ECC uncorrectable
#define cudaErrorUnsupportedLimit             215 // Unsupported limit
#define cudaErrorDeviceAlreadyInUse           216 // Device already in use
#define cudaErrorPeerAccessUnsupported        217 // Peer access unsupported
#define cudaErrorInvalidPtx                   218 // Invalid PTX
#define cudaErrorInvalidGraphicsContext       219 // Invalid graphics context
#define cudaErrorNvlinkUncorrectable          220 // Nvlink uncorrectable
#define cudaErrorJitCompilerNotFound          221 // JIT compiler not found
#define cudaErrorUnsupportedPtxVersion        222 // Unsupported PTX version
#define cudaErrorJitCompilationDisabled       223 // JIT compilation disabled
#define cudaErrorUnsupportedExecAffinity      224 // Unsupported exec affinity
#define cudaErrorUnsupportedDevSideSync       225 // Unsupported dev-side sync
#define cudaErrorInvalidSource                300 // Invalid source
#define cudaErrorFileNotFound                301 // File not found
#define cudaErrorSharedObjectSymbolNotFound  302 // Shared object symbol not found
#define cudaErrorSharedObjectInitFailed      303 // Shared object init failed
#define cudaErrorOperatingSystem             304 // Operating system error
#define cudaErrorInvalidResourceHandle       400 // Invalid resource handle
#define cudaErrorIllegalState                401 // Illegal state
#define cudaErrorLossyQuery                  402 // Lossy query
#define cudaErrorSymbolNotFound              500 // Symbol not found
#define cudaErrorNotReady                    600 // Not ready
#define cudaErrorIllegalAddress              700 // Illegal address
#define cudaErrorLaunchOutOfResources        701 // Launch out of resources
#define cudaErrorLaunchTimeout               702 // Launch timeout
#define cudaErrorLaunchIncompatibleTexturing 703 // Launch incompatible texturing
#define cudaErrorPeerAccessAlreadyEnabled    704 // Peer access already enabled
#define cudaErrorPeerAccessNotEnabled        705 // Peer access not enabled
#define cudaErrorSetOnActiveProcess          708 // Set on active process
#define cudaErrorContextIsDestroyed          709 // Context destroyed
#define cudaErrorAssert                      710 // Assert
#define cudaErrorTooManyPeers                711 // Too many peers
#define cudaErrorHostMemoryAlreadyRegistered 712 // Host memory already registered
#define cudaErrorHostMemoryNotRegistered     713 // Host memory not registered
#define cudaErrorHardwareStackError          714 // Hardware stack error
#define cudaErrorIllegalInstruction          715 // Illegal instruction
#define cudaErrorMisalignedAddress           716 // Misaligned address
#define cudaErrorInvalidAddressSpace         717 // Invalid address space
#define cudaErrorInvalidPc                   718 // Invalid PC
#define cudaErrorLaunchFailure               719 // Launch failure
#define cudaErrorCooperativeLaunchTooLarge   720 // Cooperative launch too large
#define cudaErrorNotPermitted                800 // Not permitted
#define cudaErrorNotSupported                801 // Not supported
#define cudaErrorSystemNotReady              802 // System not ready
#define cudaErrorSystemDriverMismatch        803 // Driver mismatch
#define cudaErrorCompatNotSupportedOnDevice  804 // Compat not supported
#define cudaErrorMpsConnectionFailed         805 // MPS connection failed
#define cudaErrorMpsRpcFailure               806 // MPS RPC failure
#define cudaErrorMpsServerNotReady           807 // MPS server not ready
#define cudaErrorMpsMaxClientsReached        808 // MPS max clients reached
#define cudaErrorMpsMaxConnectionsReached    809 // MPS max connections reached
#define cudaErrorMpsClientTerminated         810 // MPS client terminated
#define cudaErrorCdpNotSupported             811 // CDP not supported
#define cudaErrorCdpVersionMismatch          812 // CDP version mismatch
#define cudaErrorStreamCaptureUnsupported    900 // Stream capture unsupported
#define cudaErrorStreamCaptureInvalidated    901 // Stream capture invalidated
#define cudaErrorStreamCaptureMerge          902 // Stream capture merge
#define cudaErrorStreamCaptureUnmatched      903 // Stream capture unmatched
#define cudaErrorStreamCaptureUnjoined       904 // Stream capture unjoined
#define cudaErrorStreamCaptureIsolation      905 // Stream capture isolation
#define cudaErrorStreamCaptureImplicit       906 // Stream capture implicit
#define cudaErrorCapturedEvent               907 // Captured event
#define cudaErrorStreamCaptureWrongThread    908 // Stream capture wrong thread
#define cudaErrorTimeout                     909 // Timeout
#define cudaErrorGraphExecUpdateFailure      910 // Graph exec update failure
#define cudaErrorExternalDevice              911 // External device error
#define cudaErrorInvalidClusterSize          912 // Invalid cluster size
#define cudaErrorFunctionNotLoaded           913 // Function not loaded
#define cudaErrorInvalidResourceType         914 // Invalid resource type
#define cudaErrorInvalidResourceConfiguration 915 // Invalid resource configuration
#define cudaErrorUnknown                     999 // Unknown error
*/


// Custom function to return error string based on the error code
const char* flytcuGetErrorString(int error) {
    switch (error) {
        case cudaSuccess: return "No errors";
	case cudaErrorInvalidValue: return "Invalid value";
        case cudaErrorMemoryAllocation: return "Out of memory";
        case cudaErrorInitializationError: return "Driver not initialized";
        case cudaErrorCudartUnloading: return "CUDART unloading";
        case cudaErrorProfilerDisabled: return "Profiler disabled";
        case cudaErrorProfilerNotInitialized: return "Profiler not initialized (deprecated)";
        case cudaErrorProfilerAlreadyStarted: return "Profiler already started";
        case cudaErrorProfilerAlreadyStopped: return "Profiler already stopped";
        case cudaErrorInvalidConfiguration: return "Invalid configuration";
        case cudaErrorInvalidPitchValue: return "Invalid pitch value";
        case cudaErrorInvalidSymbol: return "Invalid symbol";
        case cudaErrorInvalidHostPointer: return "Invalid host pointer";
        case cudaErrorInvalidDevicePointer: return "Invalid device pointer";
        case cudaErrorInvalidTexture: return "Invalid texture";
        case cudaErrorInvalidTextureBinding: return "Invalid texture binding";
        case cudaErrorInvalidChannelDescriptor: return "Invalid channel descriptor";
        case cudaErrorInvalidMemcpyDirection: return "Invalid memcpy direction";
        case cudaErrorAddressOfConstant: return "Address of constant";
        case cudaErrorTextureFetchFailed: return "Texture fetch failed";
        case cudaErrorTextureNotBound: return "Texture not bound";
        case cudaErrorSynchronizationError: return "Synchronization error";
        case cudaErrorInvalidFilterSetting: return "Invalid filter setting";
        case cudaErrorInvalidNormSetting: return "Invalid norm setting";
        case cudaErrorMixedDeviceExecution: return "Mixed device execution";
        case cudaErrorNotYetImplemented: return "Not yet implemented";
        case cudaErrorMemoryValueTooLarge: return "Memory value too large";
        case cudaErrorStubLibrary: return "Stub library";
        case cudaErrorInsufficientDriver: return "Insufficient driver";
        case cudaErrorCallRequiresNewerDriver: return "Requires newer driver";
        case cudaErrorInvalidSurface: return "Invalid surface";
        case cudaErrorDuplicateVariableName: return "Duplicate variable name";
        case cudaErrorDuplicateTextureName: return "Duplicate texture name";
        case cudaErrorDuplicateSurfaceName: return "Duplicate surface name";
        case cudaErrorDevicesUnavailable: return "Devices unavailable";
        case cudaErrorIncompatibleDriverContext: return "Incompatible driver context";
        case cudaErrorMissingConfiguration: return "Missing configuration";
        case cudaErrorPriorLaunchFailure: return "Prior launch failure";
        case cudaErrorLaunchMaxDepthExceeded: return "Launch max depth exceeded";
        case cudaErrorLaunchFileScopedTex: return "Launch file scoped tex";
        case cudaErrorLaunchFileScopedSurf: return "Launch file scoped surf";
        case cudaErrorSyncDepthExceeded: return "Sync depth exceeded";
        case cudaErrorLaunchPendingCountExceeded: return "Launch pending count exceeded";
        case cudaErrorInvalidDeviceFunction: return "Invalid device function";
        case cudaErrorNoDevice: return "No device";
        case cudaErrorInvalidDevice: return "Invalid device";
        case cudaErrorDeviceNotLicensed: return "Device not licensed";
        case cudaErrorSoftwareValidityNotEstablished: return "Software validity not established";
        case cudaErrorStartupFailure: return "Startup failure";
        case cudaErrorInvalidKernelImage: return "Invalid kernel image";
        case cudaErrorDeviceUninitialized: return "Device uninitialized";
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current";
        case cudaErrorMapBufferObjectFailed: return "Map buffer object failed";
        case cudaErrorUnmapBufferObjectFailed: return "Unmap buffer object failed";
        case cudaErrorArrayIsMapped: return "Array is mapped";
        case cudaErrorAlreadyMapped: return "Already mapped";
        case cudaErrorNoKernelImageForDevice: return "No kernel image for device";
        case cudaErrorAlreadyAcquired: return "Already acquired";
        case cudaErrorNotMapped: return "Not mapped";
        case cudaErrorNotMappedAsArray: return "Not mapped as array";
        case cudaErrorNotMappedAsPointer: return "Not mapped as pointer";
        case cudaErrorECCUncorrectable: return "ECC uncorrectable";
        case cudaErrorUnsupportedLimit: return "Unsupported limit";
        case cudaErrorDeviceAlreadyInUse: return "Device already in use";
        case cudaErrorPeerAccessUnsupported: return "Peer access unsupported";
        case cudaErrorInvalidPtx: return "Invalid PTX";
        case cudaErrorInvalidGraphicsContext: return "Invalid graphics context";
        case cudaErrorNvlinkUncorrectable: return "Nvlink uncorrectable";
        case cudaErrorJitCompilerNotFound: return "JIT compiler not found";
        case cudaErrorUnsupportedPtxVersion: return "Unsupported PTX version";
        case cudaErrorJitCompilationDisabled: return "JIT compilation disabled";
        case cudaErrorUnsupportedExecAffinity: return "Unsupported exec affinity";
        case cudaErrorUnsupportedDevSideSync: return "Unsupported dev-side sync";
        case cudaErrorInvalidSource: return "Invalid source";
        case cudaErrorFileNotFound: return "File not found";
        case cudaErrorSharedObjectSymbolNotFound: return "Shared object symbol not found";
        case cudaErrorSharedObjectInitFailed: return "Shared object init failed";
        case cudaErrorOperatingSystem: return "Operating system error";
        case cudaErrorInvalidResourceHandle: return "Invalid resource handle";
        case cudaErrorIllegalState: return "Illegal state";
        case cudaErrorLossyQuery: return "Lossy query";
        case cudaErrorSymbolNotFound: return "Symbol not found";
        case cudaErrorNotReady: return "Not ready";
        case cudaErrorIllegalAddress: return "Illegal address";
        case cudaErrorLaunchOutOfResources: return "Launch out of resources";
        case cudaErrorLaunchTimeout: return "Launch timeout";
        case cudaErrorLaunchIncompatibleTexturing: return "Launch incompatible texturing";
        case cudaErrorPeerAccessAlreadyEnabled: return "Peer access already enabled";
        case cudaErrorPeerAccessNotEnabled: return "Peer access not enabled";
        case cudaErrorSetOnActiveProcess: return "Set on active process";
        case cudaErrorContextIsDestroyed: return "Context destroyed";
        case cudaErrorAssert: return "Assert triggered";
        case cudaErrorTooManyPeers: return "Too many peers";
        case cudaErrorHostMemoryAlreadyRegistered: return "Host memory already registered";
        case cudaErrorHostMemoryNotRegistered: return "Host memory not registered";
        case cudaErrorHardwareStackError: return "Hardware stack error";
        case cudaErrorIllegalInstruction: return "Illegal instruction";
        case cudaErrorMisalignedAddress: return "Misaligned address";
        case cudaErrorInvalidAddressSpace: return "Invalid address space";
        case cudaErrorInvalidPc: return "Invalid PC";
        case cudaErrorLaunchFailure: return "Launch failure";
        case cudaErrorCooperativeLaunchTooLarge: return "Cooperative launch too large";
        case cudaErrorNotPermitted: return "Not permitted";
        case cudaErrorNotSupported: return "Not supported";
        case cudaErrorSystemNotReady: return "System not ready";
        case cudaErrorSystemDriverMismatch: return "System driver mismatch";
        case cudaErrorCompatNotSupportedOnDevice: return "Compat not supported on device";
        case cudaErrorMpsConnectionFailed: return "MPS connection failed";
        case cudaErrorMpsRpcFailure: return "MPS RPC failure";
        case cudaErrorMpsServerNotReady: return "MPS server not ready";
        case cudaErrorMpsMaxClientsReached: return "MPS max clients reached";
        case cudaErrorMpsMaxConnectionsReached: return "MPS max connections reached";
        case cudaErrorMpsClientTerminated: return "MPS client terminated";
        case cudaErrorCdpNotSupported: return "CDP not supported";
        case cudaErrorCdpVersionMismatch: return "CDP version mismatch";
        case cudaErrorStreamCaptureUnsupported: return "Stream capture unsupported";
        case cudaErrorStreamCaptureInvalidated: return "Stream capture invalidated";
        case cudaErrorStreamCaptureMerge: return "Stream capture merge";
        case cudaErrorStreamCaptureUnmatched: return "Stream capture unmatched";
        case cudaErrorStreamCaptureUnjoined: return "Stream capture unjoined";
        case cudaErrorTimeout: return "Timeout";
        case cudaErrorGraphExecUpdateFailure: return "Graph exec update failure";
        case cudaErrorExternalDevice: return "External device error";
        case cudaErrorInvalidClusterSize: return "Invalid cluster size";
	case CUDA_ERROR_FUNCTION_NOT_LOADED: return "Function not loaded";
	case CUDA_ERROR_INVALID_RESOURCE_TYPE: return "Invalid resource type";
        case CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION: return "Invalid resource configuration";
        case cudaErrorUnknown: return "Unknown error";
        default: return "Unknown error code";
    }
}

/*

// Assume these are defined elsewhere, e.g., from cuDNN headers or your codebase
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_NOT_INITIALIZED 1
#define CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH 2
#define CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH 3
#define CUDNN_STATUS_DEPRECATED 4
#define CUDNN_STATUS_LICENSE_ERROR 5
#define CUDNN_STATUS_RUNTIME_IN_PROGRESS 6
#define CUDNN_STATUS_RUNTIME_FP_OVERFLOW 7
#define CUDNN_STATUS_BAD_PARAM 8
#define CUDNN_STATUS_BAD_PARAM_NULL_POINTER 9
#define CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER 10
#define CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED 11
#define CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND 12
#define CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT 13
#define CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH 14
#define CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH 15
#define CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES 16
#define CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE 17
#define CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH 18
#define CUDNN_STATUS_NOT_SUPPORTED 19
#define CUDNN_STATUS_INTERNAL_ERROR 33
#define CUDNN_STATUS_EXECUTION_FAILED 40
#define CUDNN_STATUS_UNKNOWN 10000
*/

#include <cudnn.h>
// Function to return human-readable error strings
const char* flytcudnnGetErrorString(int error) {
    switch (error) {
        case CUDNN_STATUS_SUCCESS:
            return "The operation was completed successfully.";
        case CUDNN_STATUS_NOT_INITIALIZED:
            return "The cuDNN library was not initialized properly.";
#ifdef CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
        case CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH:
#else
	case CUDNN_STATUS_VERSION_MISMATCH:
#endif
            return "Some cuDNN sub libraries have different versions, indicative of an installation issue.";
#ifdef CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH
        case CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH:
            return "The schema used for serialization is not what the current cuDNN library expects.";
        case CUDNN_STATUS_DEPRECATED:
#endif
            return "This functionality is under deprecation and will be removed in the next major version update.";
        case CUDNN_STATUS_LICENSE_ERROR:
            return "A license error was detected (license not present, expired, or misconfigured).";
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return "Some tasks in the user stream are not completed.";
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return "Numerical overflow occurred during GPU kernel execution.";
        case CUDNN_STATUS_BAD_PARAM:
            return "An incorrect value or parameter was passed to the function.";
#ifdef CUDNN_STATUS_BAD_PARAM_NULL_POINTER
        case CUDNN_STATUS_BAD_PARAM_NULL_POINTER:
            return "The cuDNN API unexpectedly received a null pointer from the user.";
        case CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER:
            return "The cuDNN API received a misaligned pointer from the user.";
        case CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED:
            return "The backend descriptor has not been finalized.";
        case CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND:
            return "The cuDNN API received an out-of-bound value.";
        case CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT:
            return "The cuDNN API received a memory buffer with insufficient space.";
        case CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH:
#else 
	case CUDNN_STATUS_ARCH_MISMATCH:
#endif
            return "The cuDNN API received an unexpected stream.";

#ifdef CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES
        case CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH:
            return "The cuDNN API received inconsistent tensor shapes.";
        case CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES:
            return "The cuDNN API received duplicated entries.";
        case CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE:
            return "The cuDNN API received an invalid or unsupported attribute type.";
        case CUDNN_STATUS_BAD_PARAM_CUDA_GRAPH_MISMATCH:
            return "The cuDNN API received an unexpected CUDA graph.";
#endif
        case CUDNN_STATUS_NOT_SUPPORTED:
            return "The functionality requested is not currently supported by cuDNN.";
#ifdef CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN
	case CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN:
	    return "cuDNN does not currently support such an operation graph pattern.";
	case CUDNN_STATUS_NOT_SUPPORTED_SHAPE:
	    return "cuDNN does not currently support the tensor shapes used in some specific operation or graph pattern.";
        case CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE:
	    return " cuDNN does not currently support the tensor data type.";
        case CUDNN_STATUS_NOT_SUPPORTED_LAYOUT:
	    return " cuDNN does not currently support the tensor layout.";
        case CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER:
	    return " The requested functionality is not compatible with the current CUDA driver.";
        case CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART:
	    return " The requested functionality is not compatible with the current CUDA runtime.";
        case CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH:
	    return " The function requires a feature absent from the current GPU device.";
        case CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING:
	    return " A runtime library required by cuDNN cannot be found in the predefined search paths. These libraries are libcuda.so (nvcuda.dll) and libnvrtc.so (nvrtc64_<Major Release Version><Minor Release Version>_0.dll and nvrtc-builtins64_<Major Release Version><Minor Release Version>.dll).";
        case CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE:
	    return " The requested functionality is not available due to missing a sublibrary.";
        case CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT:
	    return " The requested functionality is not available due to the insufficient shared memory size on the GPU.";
        case CUDNN_STATUS_NOT_SUPPORTED_PADDING:
	    return " The requested functionality is not available due to padding requirements.";
        case CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM:
	    return " The requested functionality is not available because they lead to invalid kernel launch parameters.";
        case CUDNN_STATUS_NOT_SUPPORTED_CUDA_GRAPH_NATIVE_API:
	    return " The requested functionality is not available because this particular engine does not support the native CUDA graph API. (The engines that do support that API have the behavior note CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API.)";
        case CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED:
	    return " A runtime kernel has failed to be compiled.";
        case CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE:
	    return " An unexpected internal inconsistency has been detected.";
        case CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED:
	    return " An internal host memory allocation failed inside the cuDNN library.";
        case CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED:
	    return " Resource allocation failed inside the cuDNN library.";
        case CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM:
	    return " Invalid kernel launch parameters are unexpectedly detected.";
        case CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED:
	    return " Access to GPU memory space failed, which is usually caused by a failure to bind a texture. To correct, prior to the function call, unbind any previously bound textures. Otherwise, this may indicate an internal error/bug in the library.";
        case CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER:
	    return " The GPU program failed to execute due to an error reported by the CUDA driver.";
        case CUDNN_STATUS_EXECUTION_FAILED_CUBLAS:
	    return " The GPU program failed to execute due to an error reported by cuBLAS.";
        case CUDNN_STATUS_EXECUTION_FAILED_CUDART:
	    return " The GPU program failed to execute due to an error reported by the CUDA runtime.";
        case CUDNN_STATUS_EXECUTION_FAILED_CURAND:
	    return " The GPU program failed to execute due to an error reported by cuRAND.";
#endif
        case CUDNN_STATUS_INTERNAL_ERROR:
            return "An internal cuDNN operation failed.";
        case CUDNN_STATUS_EXECUTION_FAILED:
            return "The GPU program failed to execute.";
        default:
            return "Unknown error code.";
    }
}
