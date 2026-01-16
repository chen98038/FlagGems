# MUSA (Moore Threads) Backend Configuration
message(STATUS "Configuring MUSA backend...")

set(MUSA_HOME $ENV{MUSA_HOME})
if(NOT MUSA_HOME)
    set(MUSA_HOME "/usr/local/musa")
endif()

find_library(MUSA_LIBRARY musa PATHS ${MUSA_HOME}/lib REQUIRED)

# torch_musa path
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch_musa; print(torch_musa.__path__[0])"
    OUTPUT_VARIABLE TORCH_MUSA_PATH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

function(target_link_musa_libraries target)
    target_link_libraries(${target} PRIVATE ${MUSA_LIBRARY})
    target_include_directories(${target} PRIVATE ${MUSA_HOME}/include)
    if(TORCH_MUSA_PATH)
        target_include_directories(${target} PRIVATE "${TORCH_MUSA_PATH}/include")
    endif()
endfunction()
