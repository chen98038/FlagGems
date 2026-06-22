#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"
#include "utils/autotune_helper.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor zeros(at::IntArrayRef size,
                 c10::optional<at::ScalarType> dtype,
                 c10::optional<at::Layout> layout,
                 c10::optional<at::Device> device,
                 c10::optional<bool> pin_memory) {
  int64_t n_elements = 1;
  for (auto dim : size) {
    n_elements *= dim;
  }

  auto options = at::TensorOptions()
                     .dtype(dtype.value_or(at::typeMetaToScalarType(at::get_default_dtype())))
                     .layout(layout.value_or(at::kStrided))
                     .device(device.value_or(backend::getDefaultDevice()))
                     .pinned_memory(pin_memory.value_or(false));

  TORCH_CHECK(n_elements >= 0, "Total elements must be non-negative");

  if (n_elements == 0) {
    return at::empty(size, options);
  }

  at::Tensor out = at::empty(size, options);

  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;

  const uint64_t num_blocks = (static_cast<uint64_t>(n_elements) + tile_size - 1) / tile_size;

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "zeros.py"), "zeros_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

#if defined(FLAGGEMS_USE_IX)
  f(raw_stream,
    num_blocks,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    /* num_warps = */ num_warps,
    /* num_stages = */ num_stages,
    out,
    n_elements,
    tile_size);
#else
  // ADD-CONFIG autotune: zeros_kernel in triton_src/zeros.py is decorated with
  // @libtuner(key=["n_elements"]) tuning BLOCK_SIZE + num_warps. BLOCK_SIZE is
  // SAFE to tune: the store is masked (offsets < n_elements) so any block size is
  // correct, and the 1D grid is recomputed as cdiv(n_elements, BLOCK_SIZE). The
  // tuned constexpr BLOCK_SIZE (== tile_size) is DROPPED from the lookup/launch arg
  // list (the autotuner injects it). lookup tensor count = 1 (out) MUST equal the
  // Python LibTuner.get_key dtype count = 1 (output_ptr). Kernel path is the
  // triton_src copy loaded by absolute path -- same as get_instance above.
  (void)tile_size;
  (void)num_warps;
  (void)num_stages;
  (void)num_blocks;
  static AutotunedCall ac(std::string(utils::get_triton_src_path() / "zeros.py"),
                          "zeros_kernel",
                          {"n_elements"});
  auto grid_fn = [n_elements](const triton_jit::Config &cfg) -> std::tuple<unsigned, unsigned, unsigned> {
    int64_t bs = get_int_kwarg(cfg, "BLOCK_SIZE");
    unsigned gx = static_cast<unsigned>((n_elements + bs - 1) / bs);
    return {gx, 1u, 1u};
  };
  const triton_jit::Config &cfg = ac.lookup(TuneKey {n_elements}, grid_fn, out, n_elements);
  int64_t block_size = get_int_kwarg(cfg, "BLOCK_SIZE");
  unsigned grid_x = static_cast<unsigned>((n_elements + block_size - 1) / block_size);
  f.autotuned_call(raw_stream, grid_x, 1u, 1u, cfg, out, n_elements);
#endif

  return out;
}
}  // namespace flag_gems
