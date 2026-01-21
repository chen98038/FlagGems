#pragma once

/**
 * Unified GPU Runtime API Abstraction
 *
 * This header provides backend-agnostic macros and type aliases for GPU runtime APIs.
 * It abstracts away the differences between CUDA, MUSA, and other backends.
 */

#include "flag_gems/backend/backend_config.h"

// ==============================================================================
// Backend Runtime Headers
// ==============================================================================
#if defined(BACKEND_MUSA)
    #include <musa_runtime_api.h>
#elif defined(BACKEND_NPU)
    // NPU uses ACL runtime
    #include <acl/acl.h>
#else  // CUDA, IX
    #include <cuda_runtime_api.h>
#endif

// ==============================================================================
// Device Property Types and Functions
// ==============================================================================
#if defined(BACKEND_MUSA)
    #define GPU_DEVICE_PROP              musaDeviceProp
    #define gpuGetDeviceProperties       musaGetDeviceProperties
    #define gpuGetDevice                 musaGetDevice
    #define gpuSetDevice                 musaSetDevice
    #define gpuGetDeviceCount            musaGetDeviceCount
    #define GPU_SUCCESS                  musaSuccess
    // MUSA doesn't have a version macro like CUDART_VERSION
    #define GPU_RUNTIME_VERSION          0
#elif defined(BACKEND_NPU)
    // NPU has different API structure - these are placeholders
    // Actual NPU code should use aclrtGetDeviceCount etc.
    #define GPU_DEVICE_PROP              int  // NPU doesn't have device prop struct
    #define gpuGetDeviceProperties(p, d) (0)  // No-op for NPU
    #define gpuGetDevice                 aclrtGetDevice
    #define gpuSetDevice                 aclrtSetDevice
    #define gpuGetDeviceCount            aclrtGetDeviceCount
    #define GPU_SUCCESS                  ACL_SUCCESS
    #define GPU_RUNTIME_VERSION          0
#else  // CUDA, IX
    #define GPU_DEVICE_PROP              cudaDeviceProp
    #define gpuGetDeviceProperties       cudaGetDeviceProperties
    #define gpuGetDevice                 cudaGetDevice
    #define gpuSetDevice                 cudaSetDevice
    #define gpuGetDeviceCount            cudaGetDeviceCount
    #define GPU_SUCCESS                  cudaSuccess
    #define GPU_RUNTIME_VERSION          CUDART_VERSION
#endif

// ==============================================================================
// Memory Management
// ==============================================================================
#if defined(BACKEND_MUSA)
    #define gpuMalloc                    musaMalloc
    #define gpuFree                      musaFree
    #define gpuMemcpy                    musaMemcpy
    #define gpuMemcpyAsync               musaMemcpyAsync
    #define gpuMemset                    musaMemset
    #define gpuMemsetAsync               musaMemsetAsync
    #define GPU_MEMCPY_HOST_TO_DEVICE    musaMemcpyHostToDevice
    #define GPU_MEMCPY_DEVICE_TO_HOST    musaMemcpyDeviceToHost
    #define GPU_MEMCPY_DEVICE_TO_DEVICE  musaMemcpyDeviceToDevice
#elif defined(BACKEND_NPU)
    #define gpuMalloc                    aclrtMalloc
    #define gpuFree                      aclrtFree
    #define gpuMemcpy                    aclrtMemcpy
    #define gpuMemcpyAsync               aclrtMemcpyAsync
    #define gpuMemset                    aclrtMemset
    #define gpuMemsetAsync               aclrtMemsetAsync
    #define GPU_MEMCPY_HOST_TO_DEVICE    ACL_MEMCPY_HOST_TO_DEVICE
    #define GPU_MEMCPY_DEVICE_TO_HOST    ACL_MEMCPY_DEVICE_TO_HOST
    #define GPU_MEMCPY_DEVICE_TO_DEVICE  ACL_MEMCPY_DEVICE_TO_DEVICE
#else  // CUDA, IX
    #define gpuMalloc                    cudaMalloc
    #define gpuFree                      cudaFree
    #define gpuMemcpy                    cudaMemcpy
    #define gpuMemcpyAsync               cudaMemcpyAsync
    #define gpuMemset                    cudaMemset
    #define gpuMemsetAsync               cudaMemsetAsync
    #define GPU_MEMCPY_HOST_TO_DEVICE    cudaMemcpyHostToDevice
    #define GPU_MEMCPY_DEVICE_TO_HOST    cudaMemcpyDeviceToHost
    #define GPU_MEMCPY_DEVICE_TO_DEVICE  cudaMemcpyDeviceToDevice
#endif

// ==============================================================================
// Synchronization
// ==============================================================================
#if defined(BACKEND_MUSA)
    #define gpuDeviceSynchronize         musaDeviceSynchronize
    #define gpuStreamSynchronize         musaStreamSynchronize
    #define gpuGetLastError              musaGetLastError
    #define gpuPeekAtLastError           musaPeekAtLastError
#elif defined(BACKEND_NPU)
    #define gpuDeviceSynchronize         aclrtSynchronizeDevice
    #define gpuStreamSynchronize         aclrtSynchronizeStream
    #define gpuGetLastError()            ACL_SUCCESS  // NPU has different error model
    #define gpuPeekAtLastError()         ACL_SUCCESS
#else  // CUDA, IX
    #define gpuDeviceSynchronize         cudaDeviceSynchronize
    #define gpuStreamSynchronize         cudaStreamSynchronize
    #define gpuGetLastError              cudaGetLastError
    #define gpuPeekAtLastError           cudaPeekAtLastError
#endif

// ==============================================================================
// Stream Types
// ==============================================================================
#if defined(BACKEND_MUSA)
    using gpuStream_t = musaStream_t;
    using gpuError_t = musaError_t;
#elif defined(BACKEND_NPU)
    using gpuStream_t = aclrtStream;
    using gpuError_t = aclError;
#else  // CUDA, IX
    using gpuStream_t = cudaStream_t;
    using gpuError_t = cudaError_t;
#endif

// ==============================================================================
// Event Types
// ==============================================================================
#if defined(BACKEND_MUSA)
    using gpuEvent_t = musaEvent_t;
    #define gpuEventCreate               musaEventCreate
    #define gpuEventDestroy              musaEventDestroy
    #define gpuEventRecord               musaEventRecord
    #define gpuEventSynchronize          musaEventSynchronize
    #define gpuEventElapsedTime          musaEventElapsedTime
#elif defined(BACKEND_NPU)
    using gpuEvent_t = aclrtEvent;
    #define gpuEventCreate               aclrtCreateEvent
    #define gpuEventDestroy              aclrtDestroyEvent
    #define gpuEventRecord               aclrtRecordEvent
    #define gpuEventSynchronize          aclrtSynchronizeEvent
    #define gpuEventElapsedTime          aclrtEventElapsedTime
#else  // CUDA, IX
    using gpuEvent_t = cudaEvent_t;
    #define gpuEventCreate               cudaEventCreate
    #define gpuEventDestroy              cudaEventDestroy
    #define gpuEventRecord               cudaEventRecord
    #define gpuEventSynchronize          cudaEventSynchronize
    #define gpuEventElapsedTime          cudaEventElapsedTime
#endif
