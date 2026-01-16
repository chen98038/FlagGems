#pragma once

#include "flag_gems/backend/backend_config.h"
#include <torch/torch.h>

/**
 * Unified operator implementation registration macro.
 * Automatically selects the correct Dispatch Key based on the compiled backend.
 */
#define FLAG_GEMS_LIBRARY_IMPL(ns, m) \
    TORCH_LIBRARY_IMPL(ns, FLAG_GEMS_DISPATCH_KEY, m)
