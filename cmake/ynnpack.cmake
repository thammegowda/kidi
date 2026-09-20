include_guard(GLOBAL)

set(Python3_FIND_VIRTUALENV STANDARD)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

kidi_require_submodule("Slinky" "slinky/CMakeLists.txt")
kidi_require_submodule("XNNPACK" "xnnpack/ynnpack/CMakeLists.txt")

set(SLINKY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
include_directories("${KIDI_THIRD_PARTY_DIR}/slinky")
add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/slinky"
    "${PROJECT_BINARY_DIR}/third_party/slinky"
    EXCLUDE_FROM_ALL
)

set(YNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(YNN_ENABLE_CPUINFO OFF CACHE BOOL "" FORCE)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    add_library(cpuinfo STATIC "${PROJECT_SOURCE_DIR}/src/kidi/runtime/apple_cpuinfo/cpuinfo.cpp")
    target_include_directories(cpuinfo PUBLIC "${PROJECT_SOURCE_DIR}/src/kidi/runtime/apple_cpuinfo")
    target_compile_features(cpuinfo PUBLIC cxx_std_23)
    set(YNN_ENABLE_CPUINFO ON CACHE BOOL "" FORCE)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 17)
        set(YNN_ENABLE_ARM64_SME ON CACHE BOOL "" FORCE)
        set(YNN_ENABLE_ARM64_SME2 ON CACHE BOOL "" FORCE)
    else()
        set(YNN_ENABLE_ARM64_SME OFF CACHE BOOL "" FORCE)
        set(YNN_ENABLE_ARM64_SME2 OFF CACHE BOOL "" FORCE)
    endif()
endif()

# YNNPACK's code generators currently resolve helper scripts relative to the
# root source directory. Keep that expected path as a source-tree alias.
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/ynnpack")
    file(CREATE_LINK
        "${KIDI_THIRD_PARTY_DIR}/xnnpack/ynnpack"
        "${PROJECT_SOURCE_DIR}/ynnpack"
        SYMBOLIC
        RESULT YNNPACK_SOURCE_ALIAS_RESULT
    )
    if(YNNPACK_SOURCE_ALIAS_RESULT)
        message(FATAL_ERROR "Could not create YNNPACK source alias: ${YNNPACK_SOURCE_ALIAS_RESULT}")
    endif()
endif()

add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/xnnpack/ynnpack"
    "${PROJECT_BINARY_DIR}/ynnpack"
    EXCLUDE_FROM_ALL
)

add_library(kidi_ynnpack INTERFACE)
target_link_libraries(kidi_ynnpack INTERFACE ynnpack ynnpack_composites)
add_library(kidi::ynnpack ALIAS kidi_ynnpack)