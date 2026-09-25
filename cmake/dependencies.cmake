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
kidi_require_submodule("zlib" "zlib/zlib.h")
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

# zlib's upstream CMake mutates an out-of-tree source checkout by renaming its
# checked-in zconf.h. Build the same static source set directly instead.
add_library(kidi_zlib STATIC
    "${KIDI_THIRD_PARTY_DIR}/zlib/adler32.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/compress.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/crc32.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/deflate.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/gzclose.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/gzlib.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/gzread.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/gzwrite.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/infback.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/inffast.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/inflate.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/inftrees.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/trees.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/uncompr.c"
    "${KIDI_THIRD_PARTY_DIR}/zlib/zutil.c"
)
target_include_directories(kidi_zlib PUBLIC "${KIDI_THIRD_PARTY_DIR}/zlib")
if(UNIX)
    target_compile_definitions(kidi_zlib PRIVATE HAVE_UNISTD_H)
endif()
add_library(ZLIB::ZLIB ALIAS kidi_zlib)