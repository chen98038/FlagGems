#include "flag_gems/device_info.h"
#include "flag_gems/backend/runtime_api.h"

#include <mutex>
#include <unordered_map>

namespace flag_gems::device {
namespace {
  DeviceInfo query_device(int device_id) {
    DeviceInfo info {};
    info.device_id = device_id;
    GPU_DEVICE_PROP props {};
    if (gpuGetDeviceProperties(&props, device_id) == GPU_SUCCESS) {
#if GPU_RUNTIME_VERSION >= 11020
      info.l2_cache_size = props.l2CacheSize;
#else
      info.l2_cache_size = 40ull * 1024 * 1024;
#endif
      info.sm_count = props.multiProcessorCount;
      info.major = props.major;
    } else {
      info.l2_cache_size = 40ull * 1024 * 1024;
      info.sm_count = 108;
      info.major = 8;
    }
    return info;
  }

  std::unordered_map<int, DeviceInfo> &cache() {
    static std::unordered_map<int, DeviceInfo> info_cache;
    return info_cache;
  }

  std::mutex &cache_mutex() {
    static std::mutex mutex;
    return mutex;
  }
}  // namespace

const DeviceInfo &get_device_info(int device_id) {
  {
    std::lock_guard<std::mutex> guard(cache_mutex());
    auto it = cache().find(device_id);
    if (it != cache().end()) {
      return it->second;
    }
  }
  DeviceInfo info = query_device(device_id);
  std::lock_guard<std::mutex> guard(cache_mutex());
  auto [it, inserted] = cache().emplace(device_id, info);
  if (!inserted) {
    it->second = info;
  }
  return it->second;
}

const DeviceInfo &get_current_device_info() {
  int device_id = 0;
  if (gpuGetDevice(&device_id) != GPU_SUCCESS) {
    device_id = 0;
  }
  return get_device_info(device_id);
}

int current_device_id() {
  return get_current_device_info().device_id;
}

std::size_t current_l2_cache_size() {
  return get_current_device_info().l2_cache_size;
}

int current_sm_count() {
  return get_current_device_info().sm_count;
}

int current_compute_capability_major() {
  return get_current_device_info().major;
}

}  // namespace flag_gems::device
