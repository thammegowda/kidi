cmake_minimum_required(VERSION 3.25)

project(kidi_ops LANGUAGES CXX)

option(KIDI_OPS_BUILD_HOST_TESTS "Build SDK-independent KidiOps tests" ${PROJECT_IS_TOP_LEVEL})

function(kidi_ops_add_backend name)
    set(backend "kidi_ops_${name}_backend")
    add_library(${backend} OBJECT ${ARGN})
    add_library(kidi_ops::${name}_backend ALIAS ${backend})
    target_include_directories(${backend} PRIVATE "${kidi_SOURCE_DIR}/src")
    target_compile_features(${backend} PRIVATE cxx_std_23)
    target_link_libraries(kidi PRIVATE kidi_ops::${name}_backend)
endfunction()

if(KIDI_OPS_BUILD_HOST_TESTS)
    enable_testing()
endif()
if(KIDI_ENABLE_CUDA)
    add_subdirectory(src/cuda)
endif()
if(KIDI_ENABLE_QNN OR KIDI_OPS_BUILD_HOST_TESTS)
    add_subdirectory(src/qnn)
endif()
