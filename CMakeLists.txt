cmake_minimum_required(VERSION 3.25)

project(kidi VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)

option(KIDI_BUILD_TESTS "Build kidi tests" ON)
option(KIDI_BUILD_INTEGRATION_TESTS "Register Python, browser, and model regression tests" OFF)
option(KIDI_BUILD_BENCHMARKS "Build kidi benchmarks" OFF)
option(KIDI_BUILD_PYTHON "Build the nanobind Python module" OFF)
if(EMSCRIPTEN)
    option(KIDI_WASM_THREADS "Enable WebAssembly pthreads" ON)
    option(KIDI_WASM_LARGE_MEMORY "Use 64-bit pointers lowered to wasm32 for models above 2 GiB" ON)
    option(KIDI_WASM_WEBGPU "Build the WebGPU execution backend with JSPI" OFF)
    if(KIDI_WASM_WEBGPU AND KIDI_WASM_THREADS)
        message(WARNING "WebGPU with pthreads is experimental; JSPI and pthreads are otherwise never combined")
    endif()
    add_compile_options(-msimd128)
    add_link_options(-msimd128 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=2MB)
    if(KIDI_WASM_WEBGPU)
        add_compile_options(-fwasm-exceptions)
        add_link_options(-fwasm-exceptions)
    else()
        add_compile_options(-fexceptions)
        add_link_options(-fexceptions)
    endif()
    if(KIDI_WASM_LARGE_MEMORY)
        add_compile_options(-sMEMORY64=2)
        add_link_options(-sMEMORY64=2)
    endif()
    if(KIDI_WASM_THREADS)
        add_compile_options(-pthread)
        add_link_options(-pthread -sPTHREAD_POOL_SIZE=7)
    endif()
endif()
include(cmake/dependencies.cmake)

add_library(kidi_argparse src/kidi/cli/argparse.cpp)
add_library(kidi::argparse ALIAS kidi_argparse)
target_include_directories(kidi_argparse PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(kidi_argparse PUBLIC cxx_std_23)

add_library(kidi
    src/kidi/audio/whisper.cpp
    src/kidi/checkpoint/config.cpp
    src/kidi/checkpoint/ggml/reader.cpp
    src/kidi/checkpoint/package.cpp
    src/kidi/checkpoint/prepare.cpp
    src/kidi/checkpoint/weights.cpp
    src/kidi/core/module.cpp
    src/kidi/core/version.cpp
    src/kidi/graph/graph.cpp
    src/kidi/image/gemma4.cpp
    src/kidi/inference/decoder.cpp
    src/kidi/inference/generator.cpp
    src/kidi/inference/transcriber.cpp
    src/kidi/inference/translator.cpp
    src/kidi/layers/position_encoding.cpp
    src/kidi/layers/gemma4.cpp
    src/kidi/layers/transformer.cpp
    src/kidi/layers/whisper.cpp
    src/kidi/model/gemma4.cpp
    src/kidi/model/gemma4_vision.cpp
    src/kidi/model/transformer.cpp
    src/kidi/model/whisper.cpp
    src/kidi/ops/context.cpp
    src/kidi/ops/quantization.cpp
    src/kidi/runtime/operator.cpp
    src/kidi/runtime/ynn/operators.cpp
    src/kidi/runtime/ynn/graph.cpp
    src/kidi/tensor/backend.cpp
    src/kidi/tensor/cpu_backend.cpp
    src/kidi/tensor/tensor.cpp
    src/kidi/text/tokenizer.cpp
)
add_library(kidi::kidi ALIAS kidi)

if(EMSCRIPTEN)
    if(KIDI_WASM_WEBGPU)
        target_sources(kidi PRIVATE src/kidi/tensor/web_gpu_backend.cpp src/kidi/runtime/webgpu/operators.cpp)
        target_link_libraries(kidi PRIVATE nlohmann_json::nlohmann_json)
        target_compile_definitions(kidi PUBLIC KIDI_HAS_WEBGPU=1)
        add_link_options(-sJSPI=1)
    endif()
    add_executable(kidi_wasm src/kidi/web/bindings.cpp)
    target_link_libraries(kidi_wasm PRIVATE kidi::kidi nlohmann_json::nlohmann_json)
    set_target_properties(kidi_wasm PROPERTIES OUTPUT_NAME kidi SUFFIX .mjs)
    target_link_options(kidi_wasm PRIVATE
        --no-entry -sMODULARIZE=1 -sEXPORT_NAME=createKidi -sENVIRONMENT=web,worker,node
        -sFORCE_FILESYSTEM=1 -sINITIAL_MEMORY=64MB
        "-sEXPORTED_RUNTIME_METHODS=['FS','ccall','HEAPU8']"
        "-sEXPORTED_FUNCTIONS=['_malloc','_free']"
    )
    if(KIDI_WASM_WEBGPU)
        target_link_options(kidi_wasm PRIVATE "-sJSPI_EXPORTS=['kidi_configure','kidi_load','kidi_enqueue','kidi_step','kidi_cancel']")
        if(KIDI_BUILD_TESTS)
            add_executable(kidi_webgpu_test tests/web/backend_test.cpp)
            target_link_libraries(kidi_webgpu_test PRIVATE kidi::kidi)
            set_target_properties(kidi_webgpu_test PROPERTIES SUFFIX .mjs)
            target_link_options(kidi_webgpu_test PRIVATE --no-entry -sMODULARIZE=1
                "-sEXPORTED_RUNTIME_METHODS=['HEAPU8','FS']" "-sEXPORTED_FUNCTIONS=['_malloc']"
                "-sJSPI_EXPORTS=['kidi_gpu_test','kidi_gpu_diagnose']")
        endif()
    endif()
endif()

option(KIDI_ENABLE_VULKAN "Build the Vulkan GPU backend" ${ANDROID})
option(KIDI_ENABLE_QNN "Build the Qualcomm NPU step compiler (loads QNN at runtime)" ${ANDROID})
set(KIDI_QNN_SDK "" CACHE PATH "QAIRT SDK root providing QNN headers for the NPU backend and benchmarks")
if(KIDI_ENABLE_VULKAN)
    include(cmake/vulkan.cmake)
endif()
if(KIDI_ENABLE_QNN)
    include(cmake/qnn.cmake)
endif()

if(KIDI_BUILD_BENCHMARKS)
    add_executable(kidi_gemma4_quality benchmarks/gemma4/quality.cpp)
    target_link_libraries(kidi_gemma4_quality PRIVATE kidi::kidi)
    add_executable(kidi_whisper_quality benchmarks/whisper/quality.cpp)
    target_link_libraries(kidi_whisper_quality PRIVATE kidi::kidi)
    if(UNIX)
        add_executable(kidi_android_baseline benchmarks/android/baseline.cpp)
        target_link_libraries(kidi_android_baseline PRIVATE kidi::kidi nlohmann_json::nlohmann_json)
        add_executable(kidi_npu_prefill benchmarks/android/npu_prefill.cpp)
        target_link_libraries(kidi_npu_prefill PRIVATE kidi::kidi)
    endif()
    if(ANDROID)
        find_program(KIDI_GLSLC glslc
            HINTS "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}"
            REQUIRED
        )
        set(KIDI_LOWBIT_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/android/lowbit/lowbit.comp)
        set(KIDI_LOWBIT_SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/lowbit_spirv)
        set(KIDI_LOWBIT_SPIRV)
        foreach(kernel gemv gemm)
            string(TOUPPER ${kernel} kernel_define)
            foreach(bits 8 4 2)
                set(spirv ${KIDI_LOWBIT_SPIRV_DIR}/${kernel}_bits${bits}.inc)
                add_custom_command(
                    OUTPUT ${spirv}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${KIDI_LOWBIT_SPIRV_DIR}
                    COMMAND ${KIDI_GLSLC} --target-env=vulkan1.3 -O -mfmt=num -DBITS=${bits}
                        -DKERNEL_${kernel_define}=1 ${KIDI_LOWBIT_SHADER} -o ${spirv}
                    DEPENDS ${KIDI_LOWBIT_SHADER}
                    VERBATIM
                )
                list(APPEND KIDI_LOWBIT_SPIRV ${spirv})
            endforeach()
        endforeach()
        set(KIDI_GELU_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/android/lowbit/gelu_multiply.comp)
        add_custom_command(
            OUTPUT ${KIDI_LOWBIT_SPIRV_DIR}/gelu_multiply.inc
            COMMAND ${CMAKE_COMMAND} -E make_directory ${KIDI_LOWBIT_SPIRV_DIR}
            COMMAND ${KIDI_GLSLC} --target-env=vulkan1.3 -O -mfmt=num ${KIDI_GELU_SHADER}
                -o ${KIDI_LOWBIT_SPIRV_DIR}/gelu_multiply.inc
            DEPENDS ${KIDI_GELU_SHADER}
            VERBATIM
        )
        list(APPEND KIDI_LOWBIT_SPIRV ${KIDI_LOWBIT_SPIRV_DIR}/gelu_multiply.inc)
        add_executable(kidi_lowbit_bench
            benchmarks/android/lowbit/main.cpp
            benchmarks/android/lowbit/cpu.cpp
            benchmarks/android/lowbit/gpu.cpp
            benchmarks/android/lowbit/npu.cpp
            benchmarks/android/lowbit/vulkan.cpp
            benchmarks/android/lowbit/qnn.cpp
            benchmarks/android/lowbit/ffn.cpp
            benchmarks/android/lowbit/ffn_cpu.cpp
            benchmarks/android/lowbit/ffn_gpu.cpp
            benchmarks/android/lowbit/ffn_npu.cpp
            ${KIDI_LOWBIT_SPIRV}
        )
        target_include_directories(kidi_lowbit_bench PRIVATE ${KIDI_LOWBIT_SPIRV_DIR})
        target_link_libraries(kidi_lowbit_bench PRIVATE kidi::kidi vulkan nlohmann_json::nlohmann_json)
        if(KIDI_QNN_SDK)
            target_include_directories(kidi_lowbit_bench SYSTEM PRIVATE ${KIDI_QNN_SDK}/include/QNN)
            target_compile_definitions(kidi_lowbit_bench PRIVATE KIDI_BENCH_QNN=1)
            target_link_libraries(kidi_lowbit_bench PRIVATE ${CMAKE_DL_LIBS})
        endif()
        add_executable(kidi_vulkan_probe benchmarks/android/vulkan_probe.cpp)
        target_link_libraries(kidi_vulkan_probe PRIVATE vulkan nlohmann_json::nlohmann_json)
    endif()
endif()

target_include_directories(kidi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(kidi PUBLIC cxx_std_23)
target_link_libraries(kidi
    PUBLIC
        ynnpack
        ynnpack_composites
        yaml-cpp::yaml-cpp
    PRIVATE
        tokenizerpp
        TahomaVision::Vision
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
            src/kidi/runtime/mps/eager_kernels.mm
            src/kidi/tensor/metal_backend.mm
    )
    set_source_files_properties(
        src/kidi/runtime/mps/graph.mm
        src/kidi/runtime/mps/command_batch.mm
        src/kidi/runtime/mps/quantized_linear.mm
        src/kidi/runtime/mps/eager_kernels.mm
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
        add_executable(kidi_metal_packed_bench benchmarks/metal/packed_linear_bench.mm)
        set_source_files_properties(benchmarks/metal/packed_linear_bench.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
        target_link_libraries(kidi_metal_packed_bench PRIVATE kidi::kidi ${KIDI_METAL_FRAMEWORK} ${KIDI_MPS_FRAMEWORK})
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

add_library(kidi_cli_main STATIC src/kidi/cli/main.cpp src/kidi/cli/interactive.cpp)
target_link_libraries(kidi_cli_main PUBLIC kidi::argparse kidi::kidi spdlog::spdlog nlohmann_json::nlohmann_json)

add_executable(kidi_cli src/kidi/cli/entrypoint.cpp)
set_target_properties(kidi_cli PROPERTIES OUTPUT_NAME kidi)
target_link_libraries(kidi_cli PRIVATE kidi_cli_main)

if(KIDI_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_native STABLE_ABI python/bindings.cpp)
    target_link_libraries(_native PRIVATE kidi_cli_main)
    install(TARGETS _native LIBRARY DESTINATION kidi COMPONENT python)
    install(FILES "${KIDI_THIRD_PARTY_DIR}/tokenizerspp/tools/nlcodec_to_tokenizer_json.py"
        DESTINATION kidi/converters COMPONENT python)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "from importlib.metadata import distribution; package = distribution('nanobind'); print(next(package.locate_file(file) for file in package.files if file.name == 'LICENSE' and '.dist-info' in str(file)))"
        OUTPUT_VARIABLE KIDI_NANOBIND_LICENSE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    install(FILES "${KIDI_NANOBIND_LICENSE}" DESTINATION kidi/licenses/nanobind COMPONENT python)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/kidi/checkpoint/ggml/LICENSE"
        DESTINATION kidi/licenses/ggml RENAME LICENSE COMPONENT python)
endif()

include(CTest)

if(BUILD_TESTING AND KIDI_BUILD_TESTS)
    include(cmake/tests.cmake)
endif()