cmake_minimum_required(VERSION 3.25)
project(kidi_arm_gemma4_benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT EXISTS "${EXECUTORCH_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set EXECUTORCH_SOURCE_DIR to the ExecuTorch v1.5.1 checkout")
endif()

set(EXECUTORCH_BUILD_PRESET_FILE "${EXECUTORCH_SOURCE_DIR}/tools/cmake/preset/llm.cmake"
    CACHE FILEPATH "" FORCE)
foreach(option
        EXECUTORCH_BUILD_COREML EXECUTORCH_BUILD_MPS EXECUTORCH_BUILD_CUDA
        EXECUTORCH_BUILD_PYBIND EXECUTORCH_BUILD_TESTS
        EXECUTORCH_BUILD_EXECUTOR_RUNNER EXECUTORCH_BUILD_EXTENSION_ASR_RUNNER)
    set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_XNNPACK_ENABLE_KLEIDI ON CACHE BOOL "" FORCE)
set(EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES ON CACHE BOOL "" FORCE)
set(EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_LOG_LEVEL Error CACHE STRING "" FORCE)
set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "" FORCE)
add_subdirectory("${EXECUTORCH_SOURCE_DIR}" executorch EXCLUDE_FROM_ALL)

add_executable(arm_gemma4_bench executorch_bench.cpp)
target_include_directories(arm_gemma4_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/nlohmann-json/include")
target_link_libraries(arm_gemma4_bench PRIVATE
    executorch
    executorch::backends
    extension_module extension_tensor extension_threadpool
    optimized_native_cpu_ops_lib quantized_ops_lib
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,custom_ops>")
foreach(target optimized_native_cpu_ops_lib quantized_ops_lib)
    executorch_target_link_options_shared_lib(${target})
endforeach()

if(DEFINED LLAMACPP_SOURCE_DIR AND DEFINED LLAMACPP_BUILD_DIR AND
   EXISTS "${LLAMACPP_SOURCE_DIR}/include/llama.h")
    add_executable(llamacpp_gemma4_bench llamacpp_bench.cpp)
    target_include_directories(llamacpp_gemma4_bench PRIVATE
        "${LLAMACPP_SOURCE_DIR}/include"
        "${LLAMACPP_SOURCE_DIR}/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/nlohmann-json/include")
    target_link_directories(llamacpp_gemma4_bench PRIVATE "${LLAMACPP_BUILD_DIR}/bin")
    target_link_libraries(llamacpp_gemma4_bench PRIVATE llama ggml ggml-base ggml-cpu)
    set_target_properties(llamacpp_gemma4_bench PROPERTIES
        BUILD_RPATH "${LLAMACPP_BUILD_DIR}/bin")
endif()

