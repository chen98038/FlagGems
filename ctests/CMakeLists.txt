# ==============================================================================
# Backend-specific test configuration
# ==============================================================================
if(BACKEND STREQUAL "MUSA")
    # MUSA needs: libmusa_python.so, libittnotify.so, and libtorch_python.so
    # libmusa_python.so depends on pybind11 symbols from libtorch_python.so
    # Use --allow-shlib-undefined to allow unresolved symbols from Python bindings
    set(BACKEND_EXTRA_LIBS ${MUSA_EXTRA_LIBRARIES} Torch::Torch_Python)
    link_libraries("-Wl,--allow-shlib-undefined")
else()
    set(BACKEND_EXTRA_LIBS "")
endif()

# ==============================================================================
# Helper function to add a triton test with common configuration
# Usage: add_triton_test(NAME source.cpp [EXTRA_LIBS lib1 lib2 ...])
# ==============================================================================
function(add_triton_test)
    cmake_parse_arguments(ARG "" "NAME" "EXTRA_LIBS" ${ARGN})

    # Derive source file from test name if not specified
    set(SOURCE_FILE "${ARG_NAME}.cpp")

    add_executable(${ARG_NAME} ${SOURCE_FILE})
    target_link_libraries(${ARG_NAME}
        PRIVATE
            Torch::Torch
            operators
            GTest::gtest
            GTest::gtest_main
            ${BACKEND_EXTRA_LIBS}
            ${ARG_EXTRA_LIBS}
    )
    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
endfunction()

# ==============================================================================
# Test definitions using the helper function
# ==============================================================================

# Basic operation tests
add_triton_test(NAME test_triton_pointwise)
add_triton_test(NAME test_triton_reduction)
add_triton_test(NAME test_triton_blas)
add_triton_test(NAME test_triton_norm)
add_triton_test(NAME test_triton_rope)
add_triton_test(NAME test_triton_topk)
add_triton_test(NAME test_triton_contiguous)
add_triton_test(NAME test_triton_cat)
add_triton_test(NAME test_triton_bmm)
add_triton_test(NAME test_triton_embedding)
add_triton_test(NAME test_triton_argmax)
add_triton_test(NAME test_triton_div)
add_triton_test(NAME test_triton_sort)
add_triton_test(NAME test_triton_exponential_)
add_triton_test(NAME test_triton_fill)
add_triton_test(NAME test_triton_copy)

# Tests with special source files
add_executable(test_triton_zeros_constructor test_triton_tensor_constructor.cpp)
target_link_libraries(test_triton_zeros_constructor
    PRIVATE Torch::Torch operators GTest::gtest GTest::gtest_main ${BACKEND_EXTRA_LIBS})
add_test(NAME test_triton_zeros_constructor COMMAND test_triton_zeros_constructor)

# Tests with extra dependencies
add_triton_test(NAME test_triton_softmax EXTRA_LIBS ${Python_LIBRARIES} pybind11::embed)

# Attention and KV cache tests
add_triton_test(NAME test_triton_reshape_and_cache_flash)
add_triton_test(NAME test_triton_flash_attn_varlen)

# RWKV tests
add_triton_test(NAME test_triton_rwkv_ka_fusion)
add_triton_test(NAME test_triton_rwkv_mm_sparsity)
