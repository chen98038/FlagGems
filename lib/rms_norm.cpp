#include <iostream>
#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"
#include "utils/autotune_helper.h"

namespace flag_gems {
using namespace triton_jit;

#if defined(FLAGGEMS_USE_IX)
namespace {

  int get_rms_norm_num_warps(int64_t block_size) {
    // Conservative heuristic matching default Python behavior on IX devices.
    if (block_size < 2048) {
      return 4;
    }
    if (block_size < 4096) {
      return 8;
    }
    return 16;
  }

}  // namespace
#endif

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
  at::Tensor contig_input = input.contiguous();
  at::Tensor contig_weight = weight.contiguous();
  const float epsilon_val = static_cast<float>(epsilon);
  at::IntArrayRef normalized_shape = contig_weight.sizes();
  int64_t dim = contig_input.ndimension() - normalized_shape.size();
  int64_t M = 1;
  for (int i = 0; i < dim; ++i) {
    M *= contig_input.size(i);
  }
  int64_t N = contig_input.numel() / M;

  at::Tensor out = at::empty(input.sizes(), input.options());
  at::Tensor inv_rms = at::empty({M}, at::TensorOptions().dtype(torch::kFloat32).device(input.device()));

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

#if defined(FLAGGEMS_USE_IX)
  // Original single-config path (kept for IX): rms_norm_kernel, grid (M,), BLOCK_SIZE = next_pow2(N).
  int64_t BLOCK_SIZE = utils::next_power_of_2(N);
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "rms_norm.py"),
                                      "rms_norm_kernel");
  f(raw_stream,
    M,
    1,
    1,
    get_rms_norm_num_warps(BLOCK_SIZE),
    1,
    out,
    inv_rms,
    contig_input,
    contig_weight,
    N,
    1,
    N,
    1,
    N,
    epsilon_val,
    BLOCK_SIZE);
#else
  // Autotuned loop kernel: @triton.autotune(configs=get_tuned_config("rms_norm_loop"), key=["N"]),
  // tuned TILE_N. Grid is (M,) -- one program per row, independent of TILE_N (the kernel loops over
  // N internally). The loop kernel assumes contiguous in/out (no strides in its signature).
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "rms_norm.py"),
                                      "rms_norm_loop_kernel");
  static AutotunedCall ac(std::string(utils::get_flag_gems_src_path() / "ops" / "rms_norm.py"),
                          "rms_norm_loop_kernel",
                          {"N"});
  auto grid_fn = [M](const triton_jit::Config&) -> std::tuple<unsigned, unsigned, unsigned> {
    return {static_cast<unsigned>(M), 1u, 1u};
  };
  const triton_jit::Config& cfg =
      ac.lookup(TuneKey {N}, grid_fn, out, inv_rms, contig_input, contig_weight, N, epsilon_val);
  f.autotuned_call(raw_stream,
                   static_cast<unsigned>(M),
                   1u,
                   1u,
                   cfg,
                   out,
                   inv_rms,
                   contig_input,
                   contig_weight,
                   N,
                   epsilon_val);
#endif
  return out;
}
}  // namespace flag_gems
