# NPU (Ascend) Backend Configuration
message(STATUS "Configuring NPU backend...")

set(ASCEND_HOME $ENV{ASCEND_TOOLKIT_HOME})
if(NOT ASCEND_HOME)
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

find_library(ACL_LIBRARY ascendcl PATHS ${ASCEND_HOME}/lib64 REQUIRED)

# torch_npu path
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch_npu; print(torch_npu.__path__[0])"
    OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

function(target_link_npu_libraries target)
    target_link_libraries(${target} PRIVATE ${ACL_LIBRARY})
    target_include_directories(${target} PRIVATE ${ASCEND_HOME}/include)
    if(TORCH_NPU_PATH)
        target_include_directories(${target} PRIVATE "${TORCH_NPU_PATH}/include")
    endif()
endfunction()
