#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

#if defined(BACKEND_MUSA)
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

// Backend-specific device type
#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
constexpr c10::DeviceType kBackendDevice = c10::DeviceType::PrivateUse1;
#else
constexpr c10::DeviceType kBackendDevice = c10::DeviceType::CUDA;
#endif

TEST(pointwise_op_test, add) {
  const torch::Device device(kBackendDevice, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);

  torch::Tensor out_torch = a + b;
  torch::Tensor out_triton = flag_gems::add_tensor(a, b);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}

#if defined(BACKEND_MUSA)
// Custom main to initialize Python interpreter for MUSA backend
int main(int argc, char** argv) {
  py::scoped_interpreter guard{};
  // Import torch_musa to initialize MUSA device
  py::module_::import("torch_musa");
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
