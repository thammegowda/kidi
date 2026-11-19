if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Kidi CUDA builds currently require Linux")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "Target CUDA architectures")
endif()
enable_language(CUDA)
find_package(CUDAToolkit 13 REQUIRED)

add_library(kidi_ops_cuda_runtime STATIC runtime.cu)
add_library(kidi_ops::cuda_runtime ALIAS kidi_ops_cuda_runtime)
target_include_directories(kidi_ops_cuda_runtime PUBLIC "${kidi_ops_SOURCE_DIR}/src")
target_link_libraries(kidi_ops_cuda_runtime PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(kidi_ops_cuda_runtime PROPERTIES CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)

if(TARGET kidi)
    kidi_ops_add_backend(cuda operators.cpp storage.cpp)
    target_link_libraries(kidi_ops_cuda_backend PUBLIC kidi_ops::cuda_runtime PRIVATE CUDA::cudart)
    target_compile_definitions(kidi PUBLIC KIDI_HAS_CUDA=1)
endif()

add_executable(kidi_ops_cuda_test EXCLUDE_FROM_ALL "${kidi_ops_SOURCE_DIR}/tests/cuda/runtime_test.cpp")
target_link_libraries(kidi_ops_cuda_test PRIVATE kidi_ops::cuda_runtime)
add_executable(kidi_ops_cuda_projection_benchmark EXCLUDE_FROM_ALL projection_benchmark.cpp)
target_link_libraries(kidi_ops_cuda_projection_benchmark PRIVATE kidi_ops::cuda_runtime CUDA::cudart CUDA::cublas)

set(KIDI_CUVS_ROOT "" CACHE PATH "cuVS SDK root for CUDA HNSW construction")
set(cuvs_prefix "${KIDI_CUVS_ROOT}")
if(NOT cuvs_prefix)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "import importlib.util, pathlib; s = importlib.util.find_spec('libcuvs'); print(pathlib.Path(s.origin).parent if s and s.origin else '')"
            OUTPUT_VARIABLE cuvs_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
endif()
if(cuvs_prefix)
    set(cuvs_include_hints HINTS "${cuvs_prefix}/include")
    set(cuvs_library_hints HINTS "${cuvs_prefix}/lib64" "${cuvs_prefix}/lib")
endif()
find_path(KIDI_CUVS_INCLUDE_DIR cuvs/neighbors/hnsw.h ${cuvs_include_hints})
find_path(KIDI_DLPACK_INCLUDE_DIR dlpack/dlpack.h ${cuvs_include_hints})
find_library(KIDI_CUVS_C_LIBRARY cuvs_c ${cuvs_library_hints})
if(KIDI_CUVS_INCLUDE_DIR AND KIDI_DLPACK_INCLUDE_DIR AND KIDI_CUVS_C_LIBRARY)
    message(STATUS "Found cuVS for GPU HNSW construction: ${KIDI_CUVS_C_LIBRARY}")
    find_package(Threads REQUIRED)
    # cuVS exposes a host API; avoid linking a second CUDA runtime into this adapter.
    set_source_files_properties(index.cu PROPERTIES LANGUAGE CXX)
    add_executable(kidi_index_cuda_helper index.cu)
    target_compile_features(kidi_index_cuda_helper PRIVATE cxx_std_23)
    set_target_properties(kidi_index_cuda_helper PROPERTIES
        OUTPUT_NAME kidi-index-cuda RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_include_directories(kidi_index_cuda_helper PRIVATE
        "${KIDI_CUVS_INCLUDE_DIR}" "${KIDI_DLPACK_INCLUDE_DIR}" ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(kidi_index_cuda_helper PRIVATE "${KIDI_CUVS_C_LIBRARY}" Threads::Threads)
    install(TARGETS kidi_index_cuda_helper RUNTIME DESTINATION bin)
elseif(KIDI_CUVS_ROOT)
    message(FATAL_ERROR
        "KIDI_CUVS_ROOT was set to '${KIDI_CUVS_ROOT}', but cuVS/DLPack could not be found "
        "(KIDI_CUVS_INCLUDE_DIR=${KIDI_CUVS_INCLUDE_DIR}, "
        "KIDI_DLPACK_INCLUDE_DIR=${KIDI_DLPACK_INCLUDE_DIR}, "
        "KIDI_CUVS_C_LIBRARY=${KIDI_CUVS_C_LIBRARY})")
else()
    message(STATUS "cuVS SDK not found; building CUDA inference without GPU HNSW index construction")
endif()