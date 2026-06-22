#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include <tuple>
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"
#include "utils/autotune_helper.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor contiguous(const at::Tensor &self, at::MemoryFormat memory_format) {
  TORCH_CHECK(memory_format == at::MemoryFormat::Contiguous);
  if (self.is_contiguous(memory_format = memory_format)) {
    return self;
  }
  at::Tensor out = at::empty_like(self, memory_format = memory_format);

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "contiguous.py"),
                                      "copy_kernel");

  int64_t tile_size = 1024;
  const int num_warps = 4;
  const int num_stages = 1;
  int64_t n = out.numel();
  int64_t ndim = out.dim();
  auto options = torch::TensorOptions().device(self.device()).dtype(torch::kInt64);
  at::Tensor input_sizes = torch::tensor(self.sizes(), options);
  at::Tensor input_strides = torch::tensor(self.strides(), options);
  at::Tensor out_strides = torch::tensor(out.strides(), options);
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);
#if defined(FLAGGEMS_USE_IX)
  // Original single-config path (kept for IX): copy_kernel with hardcoded tile_size.
  // operator() launches the inner @triton.jit directly and injects no config
  // kwargs, so passing tile_size positionally as BLOCK_SIZE is correct even though
  // the Python kernel now carries @libtuner.
  f(raw_stream,
    num_blocks,
    1,
    1,
    num_warps,
    num_stages,
    self,
    out,
    input_strides,
    out_strides,
    input_sizes,
    ndim,
    n,
    tile_size);
#else
  // ADD-CONFIG autotune: copy_kernel is @libtuner(key=["n_elements"]) tuning BLOCK_SIZE
  // (a flat-numel tile; kernel masks offsets < n_elements, so any BLOCK_SIZE is correct).
  // lookup() args = the kernel's non-constexpr prefix in EXACT signature order, dropping
  // the tuned BLOCK_SIZE: self,out,input_strides,out_strides,input_sizes,ndim,n.
  // Tensor count = 5 (self,out,input_strides,out_strides,input_sizes) MUST equal the
  // Python LibTuner.get_key tensor count (5). Grid = cdiv(n, BLOCK_SIZE) (tuned), so the
  // grid_fn reads BLOCK_SIZE from the chosen Config.
  static AutotunedCall ac(std::string(utils::get_triton_src_path() / "contiguous.py"),
                          "copy_kernel",
                          {"n_elements"});
  auto grid_fn = [n](const triton_jit::Config& cfg) -> std::tuple<unsigned, unsigned, unsigned> {
    int64_t bs = get_int_kwarg(cfg, "BLOCK_SIZE");
    unsigned gx = static_cast<unsigned>((n + bs - 1) / bs);
    return {gx, 1u, 1u};
  };
  const triton_jit::Config& cfg =
      ac.lookup(TuneKey {n}, grid_fn, self, out, input_strides, out_strides, input_sizes, ndim, n);
  int64_t bs = get_int_kwarg(cfg, "BLOCK_SIZE");
  unsigned gx = static_cast<unsigned>((n + bs - 1) / bs);
  f.autotuned_call(raw_stream, gx, 1u, 1u, cfg, self, out, input_strides, out_strides, input_sizes, ndim, n);
#endif
  return out;
}
}  // namespace flag_gems
