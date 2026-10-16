cmake_minimum_required(VERSION 3.25)

project(kidi_ops LANGUAGES CXX)

option(KIDI_OPS_BUILD_HOST_TESTS "Build SDK-independent KidiOps tests" ${PROJECT_IS_TOP_LEVEL})

add_library(kidi_ops_headers INTERFACE)
add_library(kidi_ops::headers ALIAS kidi_ops_headers)
target_compile_features(kidi_ops_headers INTERFACE cxx_std_17)
target_include_directories(kidi_ops_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kidi_ops/ops>
)

if(KIDI_OPS_BUILD_HOST_TESTS)
    enable_testing()
    add_executable(kidi_ops_attention_math_test tests/attention_math_test.cpp)
    target_link_libraries(kidi_ops_attention_math_test PRIVATE kidi_ops::headers)
    add_test(NAME kidi_ops_attention_math_test COMMAND kidi_ops_attention_math_test)
endif()
