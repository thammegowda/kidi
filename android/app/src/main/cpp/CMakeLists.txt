cmake_minimum_required(VERSION 3.25)

project(kidi_android LANGUAGES C CXX)

set(KIDI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KIDI_BUILD_INTEGRATION_TESTS OFF CACHE BOOL "" FORCE)
set(KIDI_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(KIDI_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../../../" "${CMAKE_BINARY_DIR}/kidi")

add_library(kidi_android SHARED bindings.cpp)
target_link_libraries(kidi_android PRIVATE kidi::kidi nlohmann_json::nlohmann_json uni-algo::uni-algo log)
target_link_options(kidi_android PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--gc-sections")