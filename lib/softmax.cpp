#include <ATen/WrapDimUtils.h>
#include <iostream>
#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"
#include "utils/autotune_helper.h"
namespace flag_gems {
using namespace triton_jit;

namespace {

  // Load Triton JIT kernel from softmax.py
  const TritonJITFunction &get_kernel(const std::string &name) {
    static const std::string src_path = (utils::get_flag_gems_src_path() / "ops" / "softmax.py").string();
    return TritonJITFunction::get_instance(src_path, name);
  }

  void compute_mnk(const at::Tensor &tensor, int dim, int64_t &M, int64_t &N, int64_t &K) {
    const auto sizes = tensor.sizes();
    M = 1;
    N = sizes[dim];
    K = 1;
    for (int i = 0; i < dim; ++i) M *= sizes[i];
    for (int i = dim + 1; i < sizes.size(); ++i) K *= sizes[i];
  }

  // Forward kernel wrapper
  at::Tensor softmax_forward(const at::Tensor &input, int dim) {
    TORCH_CHECK(input.dim() >= 2, "Softmax input must be at least 2D");

    at::Tensor output = at::empty_like(input, input.options());

    int64_t M, N, K;
    compute_mnk(input, dim, M, N, K);

    c10::DeviceGuard guard(input.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);

#if defined(FLAGGEMS_USE_IX)
    // Original hardcoded-config FORWARD path (kept for IX / non-CUDA backends):
    // single config, no SQL autotune. Untouched from the pre-migration code.
    constexpr unsigned int TILE_N = 128;
    constexpr unsigned int TILE_K = 1;
    constexpr unsigned int ONE_TILE_PER_CTA = 1;
    constexpr unsigned int NUM_WARPS = 4;
    constexpr unsigned int NUM_STAGES = 1;

    if (K == 1) {
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_inner");
      unsigned int grid_x = static_cast<unsigned int>(M);

      kernel(raw_stream, grid_x, 1, 1, NUM_WARPS, NUM_STAGES, output, input, M, N, TILE_N, ONE_TILE_PER_CTA);
    } else {
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_non_inner");
      unsigned int grid_x = static_cast<unsigned int>(M);
      unsigned int grid_y = static_cast<unsigned int>((K + TILE_K - 1) / TILE_K);

      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             input,
             M,
             N,
             K,
             TILE_N,
             TILE_K,
             ONE_TILE_PER_CTA);
    }
#else
    // Autotuned FORWARD path. Both forward kernels are now
    //   @libentry + @libtuner(configs=...fwd) + @triton.heuristics + @triton.jit.
    //   inner:     key=["M","N"]      tuned TILE_N; heuristic ONE_TILE_PER_CTA, num_warps.
    //              grid = (M, 1, 1)             -- independent of any tuned constexpr.
    //   non_inner: key=["M","N","K"]  tuned TILE_K; heuristic TILE_N(=cdiv(8192,TILE_K)),
    //              ONE_TILE_PER_CTA, num_warps. grid = (M, cdiv(K, TILE_K), 1).
    // Non-constexpr args are passed in exact kernel-signature order; the 2 tensor args
    // (output_ptr/input_ptr) give 2 dtype-key entries, matching Python get_key.
    if (K == 1) {
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_inner");
      static AutotunedCall ac_fwd_inner(
          std::string((utils::get_flag_gems_src_path() / "ops" / "softmax.py").string()),
          "softmax_kernel_inner",
          {"M", "N"});
      auto grid_fn = [M](const triton_jit::Config & /*c*/) -> std::tuple<unsigned, unsigned, unsigned> {
        return {static_cast<unsigned>(M), 1u, 1u};
      };
      const triton_jit::Config &cfg = ac_fwd_inner.lookup(TuneKey {M, N}, grid_fn, output, input, M, N);
      unsigned int grid_x = static_cast<unsigned int>(M);
      kernel.autotuned_call(raw_stream, grid_x, 1u, 1u, cfg, output, input, M, N);
    } else {
      // forward non_inner: @libtuner + @triton.heuristics trips the C++ bridge (KeyError 'TILE_N'),
      // so keep the original heuristic-based direct launch here (forward inner stays autotuned).
      const TritonJITFunction &kernel = get_kernel("softmax_kernel_non_inner");
      constexpr unsigned int TILE_N = 128, TILE_K = 1, ONE_TILE_PER_CTA = 1, NUM_WARPS = 4, NUM_STAGES = 1;
      unsigned int grid_x = static_cast<unsigned int>(M);
      unsigned int grid_y = static_cast<unsigned int>((K + TILE_K - 1) / TILE_K);
      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             input,
             M,
             N,
             K,
             TILE_N,
             TILE_K,
             ONE_TILE_PER_CTA);
    }
#endif

    return output;
  }

  // Backward kernel wrapper
  void compute_mnk_for_backward(const at::Tensor &tensor,
                                int dim,
                                int64_t &M,
                                int64_t &N,
                                int64_t &K,
                                int64_t &stride_m,
                                int64_t &stride_n,
                                int64_t &stride_k) {
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();

    M = 1;
    for (int i = 0; i < dim; ++i) M *= sizes[i];
    N = sizes[dim];
    K = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) K *= sizes[i];

    stride_m = (dim > 0) ? strides[dim - 1] : 0;
    stride_n = strides[dim];
    stride_k = (dim + 1 < sizes.size()) ? strides[dim + 1] : 1;

    if (K == 1) stride_k = 0;
    if (M == 1) stride_m = 0;
  }

  at::Tensor softmax_backward_impl(const at::Tensor &output, const at::Tensor &grad_output, int dim) {
    at::Tensor grad_output_contiguous = grad_output.contiguous();

    at::Tensor grad_input = at::empty_like(grad_output, grad_output.options());

    int64_t M, N, K;
    int64_t stride_m, stride_n, stride_k;
    compute_mnk_for_backward(output, dim, M, N, K, stride_m, stride_n, stride_k);

    constexpr unsigned int TILE_N = 128;
    constexpr unsigned int TILE_K = 1;
    constexpr unsigned int TILE_M = 64;
    constexpr unsigned int ONE_TILE_PER_CTA = 1;
    constexpr unsigned int NUM_WARPS = 4;
    constexpr unsigned int NUM_STAGES = 1;

    c10::DeviceGuard guard(output.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);

#if defined(FLAGGEMS_USE_IX)
    // Original hardcoded-config path (kept for IX / non-CUDA backends): heuristic TILE_*,
    // single config, no SQL autotune. Untouched from the pre-migration code.
    if (K == 1) {
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_inner");
      unsigned int grid_x = static_cast<unsigned int>((M + TILE_M - 1) / TILE_M);

      kernel(raw_stream,
             grid_x,
             1,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             grad_output,
             grad_input,
             M,
             N,
             TILE_M,
             TILE_N,
             ONE_TILE_PER_CTA);
    } else {
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_non_inner");
      unsigned int grid_x = static_cast<unsigned int>(M);
      unsigned int grid_y = static_cast<unsigned int>((K + TILE_K - 1) / TILE_K);

      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             grad_output,
             grad_input,
             M,
             N,
             K,
             TILE_N,
             TILE_K,
             ONE_TILE_PER_CTA);
    }
#else
    // Autotuned path. Both backward kernels are @triton.autotune + @triton.heuristics.
    //   inner:     key=["M","N"],      tuned TILE_N; heuristic TILE_M, ONE_TILE_PER_CTA.
    //              grid = (cdiv(M, TILE_M), 1, 1)            -- TILE_M comes from the heuristic,
    //              which the bridge merges into the Config, so get_int_kwarg(cfg,"TILE_M") works.
    //   non_inner: key=["M","N","K"],  tuned TILE_K; heuristic TILE_N, ONE_TILE_PER_CTA.
    //              grid = (M, cdiv(K, TILE_K), 1)            -- TILE_K is the tuned value.
    // Non-constexpr args are passed in exact kernel-signature order; the 3 tensor args
    // (out_ptr/out_grad_ptr/in_grad_ptr -> output/grad_output/grad_input) give 3 dtype-key
    // entries, matching Python LibTuner.get_key. ONE_TILE_PER_CTA is a kernel-only heuristic
    // bool (not needed for the grid) and rides through the Config automatically.
    if (K == 1) {
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_inner");
      static AutotunedCall ac_inner(
          std::string((utils::get_flag_gems_src_path() / "ops" / "softmax.py").string()),
          "softmax_backward_kernel_inner",
          {"M", "N"});
      auto grid_fn = [](const triton_jit::Config &c) -> std::tuple<unsigned, unsigned, unsigned> {
        int64_t tile_m = get_int_kwarg(c, "TILE_M");
        int64_t gm = get_int_kwarg(c, "M");
        unsigned gx = static_cast<unsigned>((gm + tile_m - 1) / tile_m);
        return {gx, 1u, 1u};
      };
      const triton_jit::Config &cfg =
          ac_inner.lookup(TuneKey {M, N}, grid_fn, output, grad_output, grad_input, M, N);
      int64_t tile_m = get_int_kwarg(cfg, "TILE_M");
      unsigned int grid_x = static_cast<unsigned int>((M + tile_m - 1) / tile_m);
      kernel.autotuned_call(raw_stream, grid_x, 1u, 1u, cfg, output, grad_output, grad_input, M, N);
    } else {
      // non_inner backward: the @triton.autotune + @triton.heuristics combo on
      // softmax_backward_kernel_non_inner trips the bridge (KeyError 'TILE_N' during resolve),
      // so keep the original heuristic-based direct launch here (inner stays autotuned).
      const TritonJITFunction &kernel = get_kernel("softmax_backward_kernel_non_inner");
      unsigned int grid_x = static_cast<unsigned int>(M);
      unsigned int grid_y = static_cast<unsigned int>((K + TILE_K - 1) / TILE_K);
      kernel(raw_stream,
             grid_x,
             grid_y,
             1,
             NUM_WARPS,
             NUM_STAGES,
             output,
             grad_output,
             grad_input,
             M,
             N,
             K,
             TILE_N,
             TILE_K,
             ONE_TILE_PER_CTA);
    }
#endif

    return grad_input;
  }

}  // namespace

// Public API
at::Tensor softmax(const at::Tensor &input, int64_t dim, bool half_to_float) {
  int64_t dim_ = at::maybe_wrap_dim(dim, input.dim());

  at::Tensor input_tensor = input;
  if (half_to_float && input.scalar_type() == at::kHalf) {
    input_tensor = input_tensor.to(at::kFloat);
  }

  at::Tensor output = softmax_forward(input_tensor, static_cast<int>(dim_));

  return output;
}

at::Tensor softmax_backward(const at::Tensor &grad_output,
                            const at::Tensor &output,
                            int64_t dim,
                            at::ScalarType input_dtype) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, output.dim());

  at::Tensor output_tensor = output;
  at::Tensor grad_output_tensor = grad_output;

  at::Tensor grad_input = softmax_backward_impl(output_tensor, grad_output_tensor, wrapped_dim);

  if (grad_input.scalar_type() != input_dtype) {
    grad_input = grad_input.to(input_dtype);
  }

  return grad_input;
}

}  // namespace flag_gems
