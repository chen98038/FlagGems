#pragma once

// Reuse triton_jit backend definitions
#include "triton_jit/backend_config.h"

namespace flag_gems {

using triton_jit::DefaultBackend;
using triton_jit::DefaultStreamType;

// Dispatch Key mapping (NPU/MUSA use PrivateUse1)
#if defined(BACKEND_NPU)
    #define FLAG_GEMS_DISPATCH_KEY PrivateUse1
#elif defined(BACKEND_MUSA)
    #define FLAG_GEMS_DISPATCH_KEY PrivateUse1
#else  // CUDA, IX
    #define FLAG_GEMS_DISPATCH_KEY CUDA
#endif

}  // namespace flag_gems
