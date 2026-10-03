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
kidi_require_submodule("Abseil" "abseil-cpp/CMakeLists.txt")
kidi_require_submodule("RE2" "re2/CMakeLists.txt")
kidi_require_submodule("tokenizerspp" "tokenizerspp/CMakeLists.txt")
kidi_require_submodule("spdlog" "spdlog/CMakeLists.txt")
kidi_require_submodule("ynnpack-dev" "ynnpack-dev/CMakeLists.txt")

add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/ynnpack-dev"
    "${PROJECT_BINARY_DIR}/third_party/ynnpack-dev"
    EXCLUDE_FROM_ALL
)

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${KIDI_THIRD_PARTY_DIR}/spdlog"
    "${PROJECT_BINARY_DIR}/third_party/spdlog"
    EXCLUDE_FROM_ALL
)

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
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(RE2_USE_ICU OFF CACHE BOOL "" FORCE)

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
FetchContent_Declare(
    abseil-cpp
    SOURCE_DIR "${KIDI_THIRD_PARTY_DIR}/abseil-cpp"
    BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/abseil-cpp"
)
FetchContent_Declare(
    re2
    SOURCE_DIR "${KIDI_THIRD_PARTY_DIR}/re2"
    BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/re2"
)

FetchContent_MakeAvailable(nlohmann_json uni_algo abseil-cpp)
if(EMSCRIPTEN)
    function(kidi_suppress_abseil_warnings directory)
        get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target IN LISTS targets)
            get_target_property(type "${target}" TYPE)
            if(type MATCHES "^(STATIC|SHARED|MODULE|OBJECT)_LIBRARY$" OR type STREQUAL "EXECUTABLE")
                target_compile_options("${target}" PRIVATE
                    "SHELL:-Wno-deprecated-pragma"
                    "SHELL:-Wno-sign-conversion"
                    "SHELL:-Wno-c++98-compat-extra-semi"
                )
            endif()
        endforeach()
        get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        foreach(subdirectory IN LISTS subdirectories)
            kidi_suppress_abseil_warnings("${subdirectory}")
        endforeach()
    endfunction()
    # Abseil enables these warnings on each target after inherited directory options.
    kidi_suppress_abseil_warnings("${KIDI_THIRD_PARTY_DIR}/abseil-cpp")
endif()
# Only the Abseil libraries RE2 links are built; the default target skips the rest.
set_property(DIRECTORY "${KIDI_THIRD_PARTY_DIR}/abseil-cpp" PROPERTY EXCLUDE_FROM_ALL TRUE)
set(_KIDI_SKIP_INSTALL_RULES "${CMAKE_SKIP_INSTALL_RULES}")
set(CMAKE_SKIP_INSTALL_RULES ON)
FetchContent_MakeAvailable(re2)
set_property(DIRECTORY "${KIDI_THIRD_PARTY_DIR}/re2" PROPERTY EXCLUDE_FROM_ALL TRUE)
set(CMAKE_SKIP_INSTALL_RULES "${_KIDI_SKIP_INSTALL_RULES}")
unset(_KIDI_SKIP_INSTALL_RULES)

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
