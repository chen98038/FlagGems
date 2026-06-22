#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"
#include "utils/autotune_helper.h"

namespace flag_gems {
using namespace triton_jit;

#if defined(FLAGGEMS_USE_IX)
namespace {

  struct AddmmLaunchConfig {
    int block_m;
    int block_n;
    int block_k;
    int num_warps;
    int num_stages;
  };

  AddmmLaunchConfig get_addmm_launch_config() {
    // Match an iluvatar tuned config instead of reusing the kunlunxin launch
    // parameters. The previous (2 warps, 5 stages) combination caused bf16
    // accuracy regressions in the C++ addmm path on IX devices.
    return {32, 64, 32, 4, 1};
  }

}  // namespace
#endif

at::Tensor addmm(const at::Tensor& self,
                 const at::Tensor& mat1,
                 const at::Tensor& mat2,
                 const at::Scalar& beta,
                 const at::Scalar& alpha) {
  at::IntArrayRef mat1_sizes = mat1.sizes();
  at::IntArrayRef mat2_sizes = mat2.sizes();
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "Incompatible dimensions");
  TORCH_CHECK(utils::broadcastable_to(self.sizes(), at::IntArrayRef({mat1_sizes[0], mat2_sizes[1]})),
              "Incompatible input shape");
  at::Tensor mat1_c = mat1.contiguous();
  // at::Tensor mat2_c = mat2.contiguous();
  at::Tensor out = at::empty({mat1_sizes[0], mat2_sizes[1]}, mat1.options());
  at::Tensor self_b = self.broadcast_to(out.sizes());
  float alpha_val = alpha.to<float>();
  float beta_val = beta.to<float>();

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "addmm.py"),
                                      "addmm_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

#if defined(FLAGGEMS_USE_IX)
  const AddmmLaunchConfig config = get_addmm_launch_config();
  unsigned int grid_x = ((mat1_sizes[0] + config.block_m - 1) / config.block_m);
  unsigned int grid_y = ((mat2_sizes[1] + config.block_n - 1) / config.block_n);
  f(/* CUstream = */ raw_stream,
    grid_x,
    grid_y,
    1,
    config.num_warps,
    config.num_stages,
    mat1_c,
    mat2,
    self_b,
    out,
    alpha_val,
    beta_val,
    mat1_sizes[0],
    mat2_sizes[1],
    mat1_sizes[1],
    mat1_c.stride(0),
    mat1_c.stride(1),
    mat2.stride(0),
    mat2.stride(1),
    self_b.stride(0),
    self_b.stride(1),
    out.stride(0),
    out.stride(1),
    config.block_m,
    config.block_n,
    config.block_k);
#else
  // addmm_kernel is @libtuner(key=["M","N","K"]); tuned BLOCK_SIZE_M/N/K.
  // IS_FP64 (constexpr, default False) is omitted -> bridge default fallback
  // (the C++ addmm path covers fp16/bf16/fp32, not fp64).
  static AutotunedCall ac(std::string(utils::get_flag_gems_src_path() / "ops" / "addmm.py"),
                          "addmm_kernel",
                          {"M", "N", "K"});

  // Grid lambda for the SQL-miss bench path. M/N and the tuned BLOCK_SIZE_*
  // all arrive through the Triton meta dict, so no C++ capture is needed.
  auto grid_fn = [](const triton_jit::Config& c) -> std::tuple<unsigned, unsigned, unsigned> {
    int64_t Mv = get_int_kwarg(c, "M");
    int64_t Nv = get_int_kwarg(c, "N");
    int64_t bm = get_int_kwarg(c, "BLOCK_SIZE_M");
    int64_t bn = get_int_kwarg(c, "BLOCK_SIZE_N");
    return {static_cast<unsigned>((Mv + bm - 1) / bm), static_cast<unsigned>((Nv + bn - 1) / bn), 1u};
  };

  const triton_jit::Config& cfg = ac.lookup(TuneKey {mat1_sizes[0], mat2_sizes[1], mat1_sizes[1]},
                                            grid_fn,
                                            mat1_c,
                                            mat2,
                                            self_b,
                                            out,
                                            alpha_val,
                                            beta_val,
                                            mat1_sizes[0],
                                            mat2_sizes[1],
                                            mat1_sizes[1],
                                            mat1_c.stride(0),
                                            mat1_c.stride(1),
                                            mat2.stride(0),
                                            mat2.stride(1),
                                            self_b.stride(0),
                                            self_b.stride(1),
                                            out.stride(0),
                                            out.stride(1));

  const int64_t bsm = get_int_kwarg(cfg, "BLOCK_SIZE_M");
  const int64_t bsn = get_int_kwarg(cfg, "BLOCK_SIZE_N");
  unsigned int grid_x = static_cast<unsigned int>((mat1_sizes[0] + bsm - 1) / bsm);
  unsigned int grid_y = static_cast<unsigned int>((mat2_sizes[1] + bsn - 1) / bsn);

  f.autotuned_call(raw_stream,
                   grid_x,
                   grid_y,
                   1u,
                   cfg,
                   mat1_c,
                   mat2,
                   self_b,
                   out,
                   alpha_val,
                   beta_val,
                   mat1_sizes[0],
                   mat2_sizes[1],
                   mat1_sizes[1],
                   mat1_c.stride(0),
                   mat1_c.stride(1),
                   mat2.stride(0),
                   mat2.stride(1),
                   self_b.stride(0),
                   self_b.stride(1),
                   out.stride(0),
                   out.stride(1));
#endif
  return out;
}

at::Tensor& addmm_out(const at::Tensor& self,
                      const at::Tensor& mat1,
                      const at::Tensor& mat2,
                      const at::Scalar& beta,
                      const at::Scalar& alpha,
                      at::Tensor& out) {
  at::IntArrayRef mat1_sizes = mat1.sizes();
  at::IntArrayRef mat2_sizes = mat2.sizes();
  TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "Incompatible dimensions");
  TORCH_CHECK(utils::broadcastable_to(self.sizes(), at::IntArrayRef({mat1_sizes[0], mat2_sizes[1]})),
              "Incompatible input shape");
  at::Tensor mat1_c = mat1.contiguous();
  // at::Tensor mat2_c = mat2.contiguous();
  at::Tensor self_b = self.broadcast_to(out.sizes());
  float alpha_val = alpha.to<float>();
  float beta_val = beta.to<float>();

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "addmm.py"),
                                      "addmm_kernel");

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

#if defined(FLAGGEMS_USE_IX)
  const AddmmLaunchConfig config = get_addmm_launch_config();
  unsigned int grid_x = ((mat1_sizes[0] + config.block_m - 1) / config.block_m);
  unsigned int grid_y = ((mat2_sizes[1] + config.block_n - 1) / config.block_n);
  f(/* CUstream = */ raw_stream,
    grid_x,
    grid_y,
    1,
    config.num_warps,
    config.num_stages,
    mat1_c,
    mat2,
    self_b,
    out,
    alpha_val,
    beta_val,
    mat1_sizes[0],
    mat2_sizes[1],
    mat1_sizes[1],
    mat1_c.stride(0),
    mat1_c.stride(1),
    mat2.stride(0),
    mat2.stride(1),
    self_b.stride(0),
    self_b.stride(1),
    out.stride(0),
    out.stride(1),
    config.block_m,
    config.block_n,
    config.block_k);
#else
  static AutotunedCall ac(std::string(utils::get_flag_gems_src_path() / "ops" / "addmm.py"),
                          "addmm_kernel",
                          {"M", "N", "K"});

  auto grid_fn = [](const triton_jit::Config& c) -> std::tuple<unsigned, unsigned, unsigned> {
    int64_t Mv = get_int_kwarg(c, "M");
    int64_t Nv = get_int_kwarg(c, "N");
    int64_t bm = get_int_kwarg(c, "BLOCK_SIZE_M");
    int64_t bn = get_int_kwarg(c, "BLOCK_SIZE_N");
    return {static_cast<unsigned>((Mv + bm - 1) / bm), static_cast<unsigned>((Nv + bn - 1) / bn), 1u};
  };

  const triton_jit::Config& cfg = ac.lookup(TuneKey {mat1_sizes[0], mat2_sizes[1], mat1_sizes[1]},
                                            grid_fn,
                                            mat1_c,
                                            mat2,
                                            self_b,
                                            out,
                                            alpha_val,
                                            beta_val,
                                            mat1_sizes[0],
                                            mat2_sizes[1],
                                            mat1_sizes[1],
                                            mat1_c.stride(0),
                                            mat1_c.stride(1),
                                            mat2.stride(0),
                                            mat2.stride(1),
                                            self_b.stride(0),
                                            self_b.stride(1),
                                            out.stride(0),
                                            out.stride(1));

  const int64_t bsm = get_int_kwarg(cfg, "BLOCK_SIZE_M");
  const int64_t bsn = get_int_kwarg(cfg, "BLOCK_SIZE_N");
  unsigned int grid_x = static_cast<unsigned int>((mat1_sizes[0] + bsm - 1) / bsm);
  unsigned int grid_y = static_cast<unsigned int>((mat2_sizes[1] + bsn - 1) / bsn);

  f.autotuned_call(raw_stream,
                   grid_x,
                   grid_y,
                   1u,
                   cfg,
                   mat1_c,
                   mat2,
                   self_b,
                   out,
                   alpha_val,
                   beta_val,
                   mat1_sizes[0],
                   mat2_sizes[1],
                   mat1_sizes[1],
                   mat1_c.stride(0),
                   mat1_c.stride(1),
                   mat2.stride(0),
                   mat2.stride(1),
                   self_b.stride(0),
                   self_b.stride(1),
                   out.stride(0),
                   out.stride(1));
#endif
  return out;
}

}  // namespace flag_gems
