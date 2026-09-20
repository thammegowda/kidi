cmake_minimum_required(VERSION 3.25)

project(kidi VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)

option(KIDI_BUILD_TESTS "Build kidi tests" ON)
option(KIDI_BUILD_BENCHMARKS "Build kidi benchmarks" OFF)
include(cmake/dependencies.cmake)
include(cmake/ynnpack.cmake)

add_library(kidi_argparse src/kidi/cli/argparse.cpp)
add_library(kidi::argparse ALIAS kidi_argparse)
target_include_directories(kidi_argparse PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(kidi_argparse PUBLIC cxx_std_23)

add_library(kidi
    src/kidi/core/module.cpp
    src/kidi/core/version.cpp
    src/kidi/inference/decoder.cpp
    src/kidi/inference/translator.cpp
    src/kidi/layers/position_encoding.cpp
    src/kidi/layers/transformer.cpp
    src/kidi/model/config.cpp
    src/kidi/model/package.cpp
    src/kidi/model/transformer.cpp
    src/kidi/model/weights.cpp
    src/kidi/ops/context.cpp
    src/kidi/runtime/operator.cpp
    src/kidi/runtime/ynn/operators.cpp
    src/kidi/runtime/ynn/graph.cpp
    src/kidi/tensor/backend.cpp
    src/kidi/tensor/cpu_backend.cpp
    src/kidi/tensor/cuda_backend.cpp
    src/kidi/tensor/tensor.cpp
    src/kidi/text/tokenizer.cpp
)
add_library(kidi::kidi ALIAS kidi)

target_include_directories(kidi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${KIDI_THIRD_PARTY_DIR}/xnnpack
)

target_compile_features(kidi PUBLIC cxx_std_23)
target_link_libraries(kidi
    PUBLIC
        kidi::ynnpack
        yaml-cpp::yaml-cpp
    PRIVATE
        tokenizerpp
        ZLIB::ZLIB
        ${CMAKE_DL_LIBS}
)

if(APPLE)
    enable_language(OBJCXX)
    find_library(KIDI_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(KIDI_METAL_FRAMEWORK Metal REQUIRED)
    find_library(KIDI_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(KIDI_MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
    target_sources(kidi
        PRIVATE
            src/kidi/runtime/mps/command_batch.mm
            src/kidi/runtime/mps/graph.mm
            src/kidi/runtime/mps/operators.cpp
            src/kidi/runtime/mps/quantized_linear.mm
            src/kidi/tensor/metal_backend.mm
    )
    set_source_files_properties(
        src/kidi/runtime/mps/graph.mm
        src/kidi/runtime/mps/command_batch.mm
        src/kidi/runtime/mps/quantized_linear.mm
        src/kidi/tensor/metal_backend.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc"
    )
    target_compile_definitions(kidi PRIVATE KIDI_HAS_METAL=1)
    target_link_libraries(kidi
        PRIVATE
            ${KIDI_FOUNDATION_FRAMEWORK}
            ${KIDI_METAL_FRAMEWORK}
            ${KIDI_MPS_FRAMEWORK}
            ${KIDI_MPSGRAPH_FRAMEWORK}
    )

    if(KIDI_BUILD_BENCHMARKS)
        add_executable(kidi_metal_int8_bench benchmarks/metal/int8_linear_bench.cpp)
        target_link_libraries(kidi_metal_int8_bench PRIVATE kidi::kidi)
        add_executable(kidi_metal_lowering_bench benchmarks/metal/lowering_bench.mm)
        set_source_files_properties(benchmarks/metal/lowering_bench.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
        target_link_libraries(kidi_metal_lowering_bench
            PRIVATE
                kidi::kidi
                ${KIDI_FOUNDATION_FRAMEWORK}
                ${KIDI_METAL_FRAMEWORK}
                ${KIDI_MPSGRAPH_FRAMEWORK}
        )
    endif()
endif()

add_executable(kidi_cli src/kidi/cli/main.cpp)
set_target_properties(kidi_cli PROPERTIES OUTPUT_NAME kidi)
target_link_libraries(kidi_cli PRIVATE kidi::argparse kidi::kidi spdlog::spdlog)

include(CTest)

if(BUILD_TESTING AND KIDI_BUILD_TESTS)
    include(cmake/tests.cmake)
endif()