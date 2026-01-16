add_library(operators
            SHARED
            device_info.cpp
            zeros.cpp
            utils.cpp
            add.cpp
            sum.cpp
            max.cpp
            mm.cpp
            rms_norm.cpp
            fused_add_rms_norm.cpp
            rotary_embedding.cpp
            topk.cpp
            addmm.cpp
            nonzero.cpp
            rotary_embedding.cpp
            contiguous.cpp
            cat.cpp
            bmm.cpp
            embedding.cpp
            argmax.cpp
            div.cpp
            softmax.cpp
            sort.cpp
	          exponential_.cpp
            fill.cpp
            reshape_and_cache_flash.cpp
            flash_attn_varlen_func.cpp
            rwkv_mm_sparsity.cpp
            rwkv_ka_fusion.cpp
            copy.cpp)

if (TRITON_GE_3P5)
  target_compile_definitions(operators PRIVATE TRITON_GE_3P5)
endif()

# Force C++20 for multi-backend libtriton_jit support (uses concepts)
set_target_properties(operators PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
target_compile_options(operators PRIVATE -std=c++20)

target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Backend conditional linking
if(BACKEND STREQUAL "CUDA" OR BACKEND STREQUAL "IX")
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit CUDA::cuda_driver)
elseif(BACKEND STREQUAL "NPU")
    target_link_npu_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
elseif(BACKEND STREQUAL "MUSA")
    target_link_musa_libraries(operators)
    target_link_libraries(operators
      PUBLIC Torch::Torch
      PRIVATE TritonJIT::triton_jit)
endif()

add_library(FlagGems::operators ALIAS operators)


if (FLAGGEMS_INSTALL)
  include(GNUInstallDirs)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/flag_gems"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS operators
    EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(EXPORT FlagGemsTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagGems
    NAMESPACE FlagGems::
    FILE FlagGemsTargets.cmake)
  export(EXPORT FlagGemsTargets FILE "${PROJECT_BINARY_DIR}/FlagGemsTargets.cmake")
endif()
