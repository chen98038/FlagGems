# ==============================================================================
# Backend-specific test configuration
# ==============================================================================
if(BACKEND STREQUAL "MUSA")
    # MUSA needs: libmusa_python.so, libittnotify.so, and libtorch_python.so
    # libmusa_python.so depends on pybind11 symbols from libtorch_python.so
    # Use --allow-shlib-undefined to allow unresolved symbols from Python bindings
    set(BACKEND_EXTRA_LIBS ${MUSA_EXTRA_LIBRARIES} Torch::Torch_Python)
    link_libraries("-Wl,--allow-shlib-undefined")

    # Get torch_musa include paths for MUSA backend
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(TORCH_MUSA_PATH "${PYTHON_SITE_PACKAGES}/torch_musa")
    get_filename_component(TORCH_MUSA_PARENT "${TORCH_MUSA_PATH}" DIRECTORY)
    set(MUSA_INCLUDE_DIRS
        "${TORCH_MUSA_PARENT}"
        "${TORCH_MUSA_PATH}/include"
        "${Python_INCLUDE_DIRS}"
    )
else()
    set(BACKEND_EXTRA_LIBS "")
    set(MUSA_INCLUDE_DIRS "")
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
    # Add backend-specific include directories
    if(MUSA_INCLUDE_DIRS)
        target_include_directories(${ARG_NAME} PRIVATE ${MUSA_INCLUDE_DIRS})
    endif()
    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
endfunction()

# ==============================================================================
# Test definitions using the helper function
# ==============================================================================

# Basic operation tests
if(BACKEND STREQUAL "MUSA")
    # MUSA tests need Python interpreter embedded
    add_triton_test(NAME test_triton_pointwise EXTRA_LIBS ${Python_LIBRARIES} pybind11::embed)
else()
    add_triton_test(NAME test_triton_pointwise)
endif()
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
if(MUSA_INCLUDE_DIRS)
    target_include_directories(test_triton_zeros_constructor PRIVATE ${MUSA_INCLUDE_DIRS})
endif()
add_test(NAME test_triton_zeros_constructor COMMAND test_triton_zeros_constructor)

# Tests with extra dependencies
add_triton_test(NAME test_triton_softmax EXTRA_LIBS ${Python_LIBRARIES} pybind11::embed)

# Attention and KV cache tests
add_triton_test(NAME test_triton_reshape_and_cache_flash)
add_triton_test(NAME test_triton_flash_attn_varlen)

# RWKV tests
add_triton_test(NAME test_triton_rwkv_ka_fusion)
add_triton_test(NAME test_triton_rwkv_mm_sparsity)
