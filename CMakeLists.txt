cmake_minimum_required(VERSION 3.25)

project(kidi VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

option(KIDI_BUILD_TESTS "Build kidi tests" ON)
include(cmake/dependencies.cmake)
include(cmake/ynnpack.cmake)

add_library(kidi_argparse src/kidi/cli/argparse.cpp)
add_library(kidi::argparse ALIAS kidi_argparse)
target_include_directories(kidi_argparse PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(kidi_argparse PUBLIC cxx_std_23)

add_library(kidi
    src/kidi/core/version.cpp
    src/kidi/model/manifest.cpp
    src/kidi/model/weights.cpp
    src/kidi/rtg/decoder.cpp
    src/kidi/rtg/embedding.cpp
    src/kidi/rtg/encoder.cpp
    src/kidi/rtg/linear.cpp
    src/kidi/rtg/package.cpp
    src/kidi/rtg/transformer_builder.cpp
    src/kidi/rtg/translator.cpp
    src/kidi/runtime/ynn.cpp
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
    PRIVATE
        tokenizerpp
        yaml-cpp::yaml-cpp
        ZLIB::ZLIB
)

add_executable(kidi_cli src/kidi/cli/main.cpp)
set_target_properties(kidi_cli PROPERTIES OUTPUT_NAME kidi)
target_link_libraries(kidi_cli PRIVATE kidi::argparse kidi::kidi)

include(CTest)

if(BUILD_TESTING AND KIDI_BUILD_TESTS)
    include(cmake/tests.cmake)
endif()