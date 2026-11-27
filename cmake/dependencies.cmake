include_guard(GLOBAL)

include(FetchContent)

set(KIDI_THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/third_party")

function(kidi_require_submodule name marker)
    if(NOT EXISTS "${KIDI_THIRD_PARTY_DIR}/${marker}")
        message(FATAL_ERROR
            "${name} is missing from third_party. "
            "Run: git submodule update --init --recursive")
    endif()
endfunction()

kidi_require_submodule("yaml-cpp" "yaml-cpp/CMakeLists.txt")
kidi_require_submodule("nlohmann-json" "nlohmann-json/CMakeLists.txt")
kidi_require_submodule("uni-algo" "uni-algo/CMakeLists.txt")
kidi_require_submodule("tokenizerspp" "tokenizerspp/CMakeLists.txt")
kidi_require_submodule("ynnpack-dev" "ynnpack-dev/CMakeLists.txt")

add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/ynnpack-dev"
    "${PROJECT_BINARY_DIR}/third_party/ynnpack-dev"
    EXCLUDE_FROM_ALL
)

if(APPLE)
    # Apple Clang can emit hidden std::exception RTTI with -fno-rtti, preventing
    # exceptions from libc++ from matching std::exception handlers in Kidi.
    target_compile_options(ynnpack_base PRIVATE -frtti)
    target_compile_options(ynnpack_composites PRIVATE -frtti)
endif()

set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/yaml-cpp"
    "${PROJECT_BINARY_DIR}/third_party/yaml-cpp"
    EXCLUDE_FROM_ALL
)

# tokenizerspp uses FetchContent when built standalone. Declare and populate
# those names first so its unchanged CMake resolves only local submodules.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
set(UNI_ALGO_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    nlohmann_json
    SOURCE_DIR "${KIDI_THIRD_PARTY_DIR}/nlohmann-json"
    BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/nlohmann-json"
)
FetchContent_Declare(
    uni_algo
    SOURCE_DIR "${KIDI_THIRD_PARTY_DIR}/uni-algo"
    BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/uni-algo"
)

FetchContent_MakeAvailable(nlohmann_json uni_algo)
target_compile_definitions(uni-algo PUBLIC
    UNI_ALGO_DISABLE_COLLATE
    UNI_ALGO_DISABLE_SEGMENT_GRAPHEME
    UNI_ALGO_DISABLE_SEGMENT_WORD
    UNI_ALGO_DISABLE_SYSTEM_LOCALE
)

set(TOKENIZERPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TOKENIZERPP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/tokenizerspp"
    "${PROJECT_BINARY_DIR}/third_party/tokenizerspp"
    EXCLUDE_FROM_ALL
)

kidi_require_submodule("tahoma-vision" "tahoma-vision/CMakeLists.txt")
set(TAHOMA_VISION_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TAHOMA_VISION_INSTALL OFF CACHE BOOL "" FORCE)
set(TAHOMA_VISION_PDF OFF CACHE BOOL "" FORCE)
set(TAHOMA_VISION_SVG OFF CACHE BOOL "" FORCE)
set(PIGZPP_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(PIGZPP_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(PIGZPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PIGZPP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(PIGZPP_INSTALL OFF CACHE BOOL "" FORCE)
set(CMAKE_PROJECT_TahomaVision_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/tahoma_vision.cmake")
add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/tahoma-vision"
    "${PROJECT_BINARY_DIR}/third_party/tahoma-vision"
    EXCLUDE_FROM_ALL
)
unset(CMAKE_PROJECT_TahomaVision_INCLUDE)

# Tahoma's pigzpp builds zlib-ng with the zlib API (ZLIB_COMPAT) for PNG decoding. Kidi uses the same library for gzip
# tokenizers: a second zlib would export the same symbols, and the linker would silently keep one for both.
add_library(kidi::zlib ALIAS zlib-ng)
