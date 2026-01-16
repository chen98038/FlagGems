#pragma once

#include "flag_gems/backend/backend_config.h"
#include <c10/core/DeviceGuard.h>
#include <ATen/core/TensorBase.h>

// Backend-specific Stream headers
#if defined(BACKEND_NPU)
    #include "torch_npu/csrc/core/npu/NPUStream.h"
#elif defined(BACKEND_MUSA)
    #include "torch_musa/csrc/core/MUSAStream.h"
#else  // CUDA, IX
    #include "c10/cuda/CUDAStream.h"
#endif

namespace flag_gems::stream {

/**
 * Get current device Stream (native type)
 */
inline DefaultStreamType getCurrentStream(c10::DeviceIndex device_index = -1) {
#if defined(BACKEND_NPU)
    auto npu_stream = c10_npu::getCurrentNPUStream(device_index);
    return static_cast<aclrtStream>(npu_stream.stream());
#elif defined(BACKEND_MUSA)
    auto musa_stream = c10_musa::getCurrentMUSAStream(device_index);
    return static_cast<musaStream_t>(musa_stream.stream());
#else  // CUDA, IX
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(device_index);
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

/**
 * Get Stream for the device of a Tensor
 */
inline DefaultStreamType getStreamForTensor(const at::Tensor& tensor) {
    c10::DeviceGuard guard(tensor.device());
    return getCurrentStream(tensor.device().index());
}

}  // namespace flag_gems::stream
