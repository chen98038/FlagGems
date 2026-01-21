#pragma once

#include "flag_gems/backend/backend_config.h"
#include <ATen/core/TensorBase.h>
#include <c10/core/DeviceType.h>

namespace flag_gems {

/**
 * Get the DeviceType for the current compiled backend.
 */
inline constexpr c10::DeviceType getBackendDeviceType() {
#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    return c10::DeviceType::PrivateUse1;
#else  // CUDA, IX
    return c10::DeviceType::CUDA;
#endif
}

/**
 * Check if a tensor is on the current backend's device.
 * This is the backend-agnostic replacement for is_cuda().
 */
inline bool isBackendDevice(const at::Tensor& tensor) {
    return tensor.device().type() == getBackendDeviceType();
}

/**
 * Check if a tensor is on the current backend's device (alternative overload).
 */
inline bool isBackendDevice(const at::TensorBase& tensor) {
    return tensor.device().type() == getBackendDeviceType();
}

/**
 * Get current device index for the backend.
 */
inline c10::DeviceIndex getCurrentDeviceIndex() {
#if defined(BACKEND_NPU)
    // NPU: use torch_npu API
    return c10_npu::current_device();
#elif defined(BACKEND_MUSA)
    // MUSA: use torch_musa API
    return c10::musa::current_device();
#else  // CUDA, IX
    return at::cuda::current_device();
#endif
}

/**
 * Get the current backend device.
 */
inline at::Device getCurrentBackendDevice() {
    return at::Device(getBackendDeviceType(), getCurrentDeviceIndex());
}

/**
 * Get backend device type name for error messages.
 */
inline constexpr const char* getBackendDeviceName() {
#if defined(BACKEND_NPU)
    return "NPU";
#elif defined(BACKEND_MUSA)
    return "MUSA";
#elif defined(BACKEND_IX)
    return "IX";
#else  // CUDA
    return "CUDA";
#endif
}

}  // namespace flag_gems
