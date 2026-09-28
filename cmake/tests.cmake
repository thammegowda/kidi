include_guard(GLOBAL)

function(kidi_add_test target source)
    add_executable(${target} "${source}")
    target_link_libraries(${target} PRIVATE kidi::kidi ${ARGN})
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS native)
endfunction()

add_executable(kidi_argparse_test tests/kidi/cli/argparse_test.cpp)
target_link_libraries(kidi_argparse_test PRIVATE kidi::argparse)
add_test(NAME kidi_argparse_test COMMAND kidi_argparse_test)
set_tests_properties(kidi_argparse_test PROPERTIES LABELS native)

kidi_add_test(kidi_manifest_test tests/kidi/model/config_test.cpp)
kidi_add_test(kidi_whisper_audio_test tests/kidi/audio/whisper_test.cpp)
kidi_add_test(kidi_gemma4_image_test tests/kidi/image/gemma4_test.cpp TahomaVision::Vision)
kidi_add_test(kidi_tokenizer_test tests/kidi/text/tokenizer_test.cpp ZLIB::ZLIB)
kidi_add_test(kidi_rtg_package_test tests/kidi/model/package_test.cpp)
kidi_add_test(kidi_gemma4_test tests/kidi/model/gemma4_test.cpp)
target_compile_definitions(kidi_gemma4_test PRIVATE KIDI_GEMMA4_FIXTURE="${PROJECT_SOURCE_DIR}/tests/data/gemma4")
kidi_add_test(kidi_whisper_test tests/kidi/model/whisper_test.cpp)
kidi_add_test(kidi_decoder_test tests/kidi/inference/decoder_test.cpp)
kidi_add_test(kidi_weights_test tests/kidi/model/weights_test.cpp)
kidi_add_test(kidi_tensor_test tests/kidi/tensor/tensor_test.cpp)
kidi_add_test(kidi_eager_test tests/kidi/layers/eager_test.cpp)
kidi_add_test(kidi_ynnpack_weights_test tests/kidi/runtime/ynn/weights_test.cpp)
kidi_add_test(kidi_ynn_runtime_test tests/kidi/runtime/ynn/graph_test.cpp)
kidi_add_test(kidi_rtg_embedding_test tests/kidi/layers/embedding_test.cpp)
kidi_add_test(kidi_rtg_transformer_builder_test tests/kidi/layers/transformer_test.cpp)
kidi_add_test(kidi_rtg_attention_test tests/kidi/layers/attention_test.cpp)
kidi_add_test(kidi_rtg_precision_test tests/kidi/layers/precision_test.cpp)
kidi_add_test(kidi_rtg_int8_precision_test tests/kidi/layers/int8_precision_test.cpp)

if(KIDI_BUILD_INTEGRATION_TESTS)
    if(NOT KIDI_TEST_PYTHON OR NOT EXISTS "${KIDI_TEST_PYTHON}")
        message(FATAL_ERROR "KIDI_TEST_PYTHON must name the prepared test-environment interpreter")
    endif()
    if(NOT KIDI_TEST_BINARY OR NOT EXISTS "${KIDI_TEST_BINARY}")
        message(FATAL_ERROR "KIDI_TEST_BINARY must name the release CLI")
    endif()
    find_program(KIDI_TEST_NODE node REQUIRED)

    add_test(NAME kidi_python_test COMMAND "${KIDI_TEST_PYTHON}" -m unittest discover
        -s "${PROJECT_SOURCE_DIR}/tests" -p python_cli_test.py)
    set_tests_properties(kidi_python_test PROPERTIES LABELS python WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

    add_test(NAME kidi_browser_test COMMAND "${KIDI_TEST_NODE}" --test
        "${PROJECT_SOURCE_DIR}/tests/web/model_cache_test.mjs"
        "${PROJECT_SOURCE_DIR}/tests/web/speech_test.mjs")
    set_tests_properties(kidi_browser_test PROPERTIES LABELS browser WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

    add_test(NAME kidi_rtg_regression_test COMMAND "${KIDI_TEST_PYTHON}"
        "${PROJECT_SOURCE_DIR}/tests/rtg_regression.py" run --binary "${KIDI_TEST_BINARY}"
        --backend "${KIDI_TEST_BACKEND}" --threads "${KIDI_TEST_THREADS}" --min-chrf "${KIDI_TEST_MIN_CHRF}")
    set_tests_properties(kidi_rtg_regression_test PROPERTIES LABELS regression WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()
