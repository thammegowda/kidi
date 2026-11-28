# Reuse the GoogleTest pinned by ynnpack-dev instead of vendoring a second copy.
if(NOT TARGET gtest)
    kidi_require_submodule("googletest" "ynnpack-dev/third_party/googletest/CMakeLists.txt")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(
        "${KIDI_THIRD_PARTY_DIR}/ynnpack-dev/third_party/googletest"
        "${PROJECT_BINARY_DIR}/third_party/googletest"
        EXCLUDE_FROM_ALL
    )
endif()
include(GoogleTest)

set(KIDI_GPU_LAUNCHER sh "${PROJECT_SOURCE_DIR}/cmake/gpu-launch.sh")
# CTest 3.28+ runs this first and schedules GPU tests from the pool it writes, sized to the
# GPUs visible at test time; tests with RESOURCE_GROUPS must require the `gpu_pool` fixture.
set(KIDI_GPU_POOL "${CMAKE_CURRENT_BINARY_DIR}/ctest-resources.json")
add_test(NAME GpuPool COMMAND ${KIDI_GPU_LAUNCHER} --spec "${KIDI_GPU_POOL}")
set_tests_properties(GpuPool PROPERTIES
    LABELS native
    FIXTURES_SETUP gpu_pool
    GENERATED_RESOURCE_SPEC_FILE "${KIDI_GPU_POOL}")

# Registers every TEST() in an existing GoogleTest executable with CTest as `<Suite>.<Name>`.
# Names stay short so CTest status lines fit one terminal row; select a binary by suite, e.g.
# `ctest -R '^Gemma4'`. Keep suite names unique across binaries and full names <= 40 chars.
function(kidi_discover_tests target)
    cmake_parse_arguments(TEST "" "COST;RESOURCE_LOCK;TIMEOUT" "LABELS" ${ARGN})
    if(NOT TEST_LABELS)
        set(TEST_LABELS native)
    endif()
    list(JOIN TEST_LABELS "\\\\;" labels)
    set(properties LABELS "${labels}")
    # `cuda` tests each take one GPU from the `gpus` pool written by GpuPool, and
    # cmake/gpu-launch.sh pins the test to it, so concurrent GPU tests land on different GPUs.
    if("cuda" IN_LIST TEST_LABELS)
        list(APPEND properties RESOURCE_GROUPS gpus:1 FIXTURES_REQUIRED gpu_pool)
        set_property(TARGET ${target} PROPERTY TEST_LAUNCHER ${KIDI_GPU_LAUNCHER})
    endif()
    if(TEST_COST)
        list(APPEND properties COST ${TEST_COST})
    endif()
    if(TEST_RESOURCE_LOCK)
        list(APPEND properties RESOURCE_LOCK ${TEST_RESOURCE_LOCK})
    endif()
    if(TEST_TIMEOUT)
        list(APPEND properties TIMEOUT ${TEST_TIMEOUT})
    endif()
    # Parameter generators already name cases (e.g. `/qat`); skip CMake's `# GetParam() = ...` suffix.
    gtest_discover_tests(${target}
        NO_PRETTY_TYPES NO_PRETTY_VALUES
        DISCOVERY_MODE PRE_TEST
        PROPERTIES ${properties})
endfunction()

# kidi_add_test(<target> <source> [STANDALONE] [LINK libs...] [LABELS labels...] [COST n] [TIMEOUT seconds])
# Builds a GoogleTest executable linked against kidi::kidi (unless STANDALONE).
function(kidi_add_test target source)
    cmake_parse_arguments(TEST "STANDALONE" "COST;RESOURCE_LOCK;TIMEOUT" "LINK;LABELS" ${ARGN})
    if(NOT IS_ABSOLUTE "${source}")
        set(source "${PROJECT_SOURCE_DIR}/${source}")
    endif()
    add_executable(${target} "${source}")
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    if(NOT TEST_STANDALONE)
        target_link_libraries(${target} PRIVATE kidi::kidi)
    endif()
    target_link_libraries(${target} PRIVATE GTest::gtest_main ${TEST_LINK})
    set(discover_args LABELS ${TEST_LABELS})
    if(TEST_COST)
        list(APPEND discover_args COST ${TEST_COST})
    endif()
    if(TEST_RESOURCE_LOCK)
        list(APPEND discover_args RESOURCE_LOCK ${TEST_RESOURCE_LOCK})
    endif()
    if(TEST_TIMEOUT)
        list(APPEND discover_args TIMEOUT ${TEST_TIMEOUT})
    endif()
    kidi_discover_tests(${target} ${discover_args})
endfunction()

kidi_add_test(kidi_argparse_test tests/kidi/cli/argparse_test.cpp STANDALONE LINK kidi::argparse)
if(TARGET kidi::agent)
    kidi_add_test(kidi_agent_test kidi-agent/tests/agent_test.cpp STANDALONE LINK kidi::agent)
    target_sources(kidi_agent_test PRIVATE "${PROJECT_SOURCE_DIR}/kidi-agent/tests/workspace_test.cpp")
    target_sources(kidi_agent_test PRIVATE "${PROJECT_SOURCE_DIR}/kidi-agent/tests/config_test.cpp")
    if(TARGET kidi_agent_command AND UNIX)
        kidi_add_test(kidi_agent_command_test kidi-agent/tests/command_test.cpp
            STANDALONE LINK kidi_agent_command TIMEOUT 60)
    endif()
    if(TARGET kidi::agent_http)
        kidi_add_test(kidi_agent_chat_client_test kidi-agent/tests/chat_client_test.cpp
            STANDALONE LINK kidi::agent_http liteapp::httplib)
    endif()
endif()

if(KIDI_BUILD_AGENT AND KIDI_AGENT_PLATFORM AND NOT CMAKE_CROSSCOMPILING)
    find_program(KIDI_TEST_BASH bash)
    find_program(KIDI_TEST_CURL curl)
    if(KIDI_TEST_BASH AND KIDI_TEST_CURL)
        add_test(NAME kidi_agent_managed
            COMMAND "${KIDI_TEST_BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/kidi/cli/managed_agent_test.sh"
                "$<TARGET_FILE:kidi_cli>" "${PROJECT_SOURCE_DIR}/tests/data/gemma4" "${KIDI_TEST_CURL}")
        set_tests_properties(kidi_agent_managed PROPERTIES TIMEOUT 30 LABELS native)
    endif()
endif()

kidi_add_test(kidi_editor_test tests/kidi/app/editor_test.cpp)
kidi_add_test(kidi_prompts_test tests/kidi/app/prompts_test.cpp)
target_compile_definitions(kidi_prompts_test PRIVATE KIDI_APPLICATION_CONFIG="${PROJECT_SOURCE_DIR}/examples/kidi-app.yaml")
kidi_add_test(kidi_document_test tests/kidi/app/document_test.cpp)
kidi_add_test(kidi_manifest_test tests/kidi/checkpoint/config_test.cpp)
kidi_add_test(kidi_hub_test tests/kidi/checkpoint/hub_test.cpp)
kidi_add_test(kidi_whisper_audio_test tests/kidi/audio/whisper_test.cpp COST 50)
kidi_add_test(kidi_duration_test tests/kidi/audio/duration_test.cpp)
kidi_add_test(kidi_wav_test tests/kidi/audio/wav_test.cpp)
kidi_add_test(kidi_gemma4_image_test tests/kidi/image/gemma4_test.cpp LINK TahomaVision::Vision)
kidi_add_test(kidi_response_test tests/kidi/text/response_test.cpp)
kidi_add_test(kidi_tokenizer_test tests/kidi/text/tokenizer_test.cpp LINK kidi::zlib)
target_compile_definitions(kidi_tokenizer_test PRIVATE
    KIDI_INKLING_TEMPLATE="${PROJECT_SOURCE_DIR}/third_party/tokenizerspp/tests/chat-templates/inkling2.jinja")
kidi_add_test(kidi_phonemizer_test tests/kidi/text/phonemizer_test.cpp)
kidi_add_test(kidi_sentence_test tests/kidi/text/sentence_test.cpp)
kidi_add_test(kidi_rtg_package_test tests/kidi/checkpoint/package_test.cpp)
set(gemma4_test_labels native regression)
if(KIDI_ENABLE_CUDA)
    list(APPEND gemma4_test_labels cuda)
endif()
kidi_add_test(kidi_gemma4_test tests/kidi/model/gemma4_test.cpp COST 100 LABELS ${gemma4_test_labels})
if(KIDI_ENABLE_CUDA)
    # These checkpoints do not fit together on the single-GPU two-slot fallback.
    kidi_add_test(kidi_inkling2_test tests/kidi/model/inkling2_test.cpp LABELS native cuda RESOURCE_LOCK inkling2_model)
endif()
if(KIDI_BUILD_INTEGRATION_TESTS)
    # Real-checkpoint CPU INT8 regression (~26 GB cache, 64 GiB of RAM); kept out of the fast `make test` loop.
    kidi_add_test(kidi_inkling2_cpu_test tests/kidi/model/inkling2_cpu_test.cpp RESOURCE_LOCK inkling2_model)
    # Stable source code for the teacher-forced prefill quality check.
    target_compile_definitions(kidi_inkling2_cpu_test PRIVATE
        KIDI_INKLING2_TEXT_FIXTURE="${KIDI_THIRD_PARTY_DIR}/ynnpack-dev/third_party/googletest/googletest/src/gtest.cc")
endif()
kidi_add_test(kidi_kokoro_test tests/kidi/model/kokoro_test.cpp)
kidi_add_test(kidi_model_registry_test tests/kidi/model/registry_test.cpp)
target_compile_definitions(kidi_gemma4_test PRIVATE KIDI_GEMMA4_FIXTURE="${PROJECT_SOURCE_DIR}/tests/data/gemma4")
kidi_add_test(kidi_whisper_test tests/kidi/model/whisper_test.cpp)
kidi_add_test(kidi_decoder_test tests/kidi/inference/decoder_test.cpp)
kidi_add_test(kidi_chat_api_test tests/kidi/server/chat_api_test.cpp LINK kidi_server)
kidi_add_test(kidi_chat_test tests/kidi/inference/chat_test.cpp LINK kidi_cli_main liteapp::http liteapp::httplib)
target_sources(kidi_chat_test PRIVATE "${PROJECT_SOURCE_DIR}/tests/kidi/cli/backend_test.cpp")
target_compile_definitions(kidi_chat_test PRIVATE KIDI_GEMMA4_FIXTURE="${PROJECT_SOURCE_DIR}/tests/data/gemma4")
kidi_add_test(kidi_speculative_test tests/kidi/inference/speculative_test.cpp)
kidi_add_test(kidi_gemma4_mtp_test tests/kidi/model/gemma4_mtp_test.cpp)
target_compile_definitions(kidi_gemma4_mtp_test PRIVATE KIDI_GEMMA4_FIXTURE="${PROJECT_SOURCE_DIR}/tests/data/gemma4")
kidi_add_test(kidi_gemma4_assistant_test tests/kidi/model/gemma4_assistant_test.cpp)
target_compile_definitions(kidi_gemma4_assistant_test PRIVATE KIDI_GEMMA4_FIXTURE="${PROJECT_SOURCE_DIR}/tests/data/gemma4")
kidi_add_test(kidi_voice_test tests/kidi/inference/voice_test.cpp)
kidi_add_test(kidi_weights_test tests/kidi/checkpoint/weights_test.cpp)
kidi_add_test(kidi_tensor_test tests/kidi/tensor/tensor_test.cpp)
if(TARGET kidi_ops::cuda_runtime)
    kidi_add_test(kidi_cuda_test kidi-ops/tests/cuda/operators_test.cpp LINK kidi_ops::cuda_runtime LABELS native cuda)
    set_target_properties(kidi_ops_cuda_test PROPERTIES
        EXCLUDE_FROM_ALL OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(kidi_ops_cuda_test PRIVATE GTest::gtest_main)
    kidi_discover_tests(kidi_ops_cuda_test LABELS native cuda)
    # Per-case tests are pinned to one GPU; these runs keep two GPUs visible to cover
    # per-device state such as kernel attributes and cross-device routing.
    foreach(binary kidi_cuda_test kidi_ops_cuda_test)
        string(REGEX REPLACE "^kidi_(.*)_test$" "MultiGpu.\\1" name ${binary})
        add_test(NAME ${name} COMMAND ${KIDI_GPU_LAUNCHER} $<TARGET_FILE:${binary}>)
        set_tests_properties(${name} PROPERTIES LABELS "native;cuda" RESOURCE_GROUPS "2,gpus:1" FIXTURES_REQUIRED gpu_pool)
    endforeach()
endif()
kidi_add_test(kidi_eager_test tests/kidi/layers/eager_test.cpp)
kidi_add_test(kidi_fused_cpu_test tests/kidi/ops/fused_cpu_test.cpp)
if(KIDI_ENABLE_CUDA)
    target_link_libraries(kidi_fused_cpu_test PRIVATE kidi_ops::cuda_runtime)
endif()
if(APPLE)
    kidi_add_test(kidi_fused_metal_test tests/kidi/ops/fused_metal_test.cpp)
endif()
kidi_add_test(kidi_audio_layers_test tests/kidi/layers/audio_test.cpp)
kidi_add_test(kidi_graph_test tests/kidi/graph/graph_test.cpp)
kidi_add_test(kidi_hnsw_test tests/kidi/search/hnsw_test.cpp)
kidi_add_test(kidi_index_test tests/kidi/cli/index_test.cpp LINK kidi_cli_main)
target_compile_definitions(kidi_index_test PRIVATE KIDI_GEMMA4_FIXTURE="${PROJECT_SOURCE_DIR}/tests/data/gemma4")
kidi_add_test(kidi_ynnpack_weights_test tests/kidi/runtime/ynn/weights_test.cpp)
kidi_add_test(kidi_ynn_runtime_test tests/kidi/runtime/ynn/graph_test.cpp)
kidi_add_test(kidi_ynn_numa_test tests/kidi/runtime/ynn/numa_test.cpp)
kidi_add_test(kidi_ynn_worker_pool_test tests/kidi/runtime/ynn/worker_pool_test.cpp)
kidi_add_test(kidi_qnn_step_compiler_test kidi-ops/tests/qnn/compiler_test.cpp)
if(KIDI_ENABLE_VULKAN)
    kidi_add_test(kidi_vulkan_operator_test tests/kidi/runtime/vk/operators_test.cpp)
endif()
kidi_add_test(kidi_rtg_embedding_test tests/kidi/layers/embedding_test.cpp)
kidi_add_test(kidi_rtg_transformer_builder_test tests/kidi/layers/transformer_test.cpp)
kidi_add_test(kidi_rtg_attention_test tests/kidi/layers/attention_test.cpp)
kidi_add_test(kidi_rtg_precision_test tests/kidi/layers/precision_test.cpp)
kidi_add_test(kidi_rtg_int8_precision_test tests/kidi/layers/int8_precision_test.cpp)
kidi_add_test(kidi_qwen3_test tests/kidi/model/qwen3_test.cpp)
if(KIDI_BUILD_INTEGRATION_TESTS)
    if(NOT KIDI_TEST_PYTHON OR NOT EXISTS "${KIDI_TEST_PYTHON}")
        message(FATAL_ERROR "KIDI_TEST_PYTHON must name the prepared test-environment interpreter")
    endif()
    add_test(NAME kidi_python_test COMMAND "${KIDI_TEST_PYTHON}" -m unittest discover
        -s "${PROJECT_SOURCE_DIR}/tests" -p "*_test.py")
    set_tests_properties(kidi_python_test PROPERTIES
        LABELS python
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        ENVIRONMENT "KIDI_BIN=$<TARGET_FILE:kidi_cli>;KIDI_TEST_BACKEND=${KIDI_TEST_BACKEND}")

    if(NOT KIDI_ENABLE_CUDA)
        add_test(NAME kidi_rtg_regression_test COMMAND "${KIDI_TEST_PYTHON}"
            "${PROJECT_SOURCE_DIR}/tests/rtg_regression.py" run --binary "$<TARGET_FILE:kidi_cli>"
            --backend "${KIDI_TEST_BACKEND}" --threads "${KIDI_TEST_THREADS}" --min-chrf "${KIDI_TEST_MIN_CHRF}")
        set_tests_properties(kidi_rtg_regression_test PROPERTIES LABELS regression WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    else()
        message(STATUS "RTG regression excluded: RTG does not support the CUDA backend")
    endif()
endif()
