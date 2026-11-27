# Vulkan GPU backend: sources, embedded SPIR-V kernels, and KIDI_HAS_VULKAN.
find_program(KIDI_VULKAN_GLSLC glslc HINTS "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}" REQUIRED)
set(KIDI_VULKAN_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan")
file(MAKE_DIRECTORY "${KIDI_VULKAN_GENERATED}")
set(KIDI_VULKAN_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/src/kidi/runtime/vulkan/packed_linear.comp")
set(KIDI_VULKAN_GELU_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/src/kidi/runtime/vulkan/gelu_multiply.comp")
set(KIDI_VULKAN_SPIRV_INCS)
foreach(kind IN ITEMS gemv gemm)
    foreach(bits IN ITEMS 2 4 8)
        set(output "${KIDI_VULKAN_GENERATED}/packed_${kind}_bits${bits}.inc")
        set(output_i8 "${KIDI_VULKAN_GENERATED}/packed_${kind}_i8_bits${bits}.inc")
        string(TOUPPER "${kind}" upper_kind)
        add_custom_command(
            OUTPUT "${output}"
            COMMAND "${KIDI_VULKAN_GLSLC}" --target-env=vulkan1.3 -O -mfmt=num
                    "-DKERNEL_${upper_kind}" "-DBITS=${bits}" "${KIDI_VULKAN_SHADER}" -o "${output}"
            DEPENDS "${KIDI_VULKAN_SHADER}"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${output_i8}"
            COMMAND "${KIDI_VULKAN_GLSLC}" --target-env=vulkan1.3 -O -mfmt=num
                    "-DKERNEL_${upper_kind}" "-DBITS=${bits}" -DOUTPUT_I8 "${KIDI_VULKAN_SHADER}" -o "${output_i8}"
            DEPENDS "${KIDI_VULKAN_SHADER}"
            VERBATIM
        )
        list(APPEND KIDI_VULKAN_SPIRV_INCS "${output}")
        list(APPEND KIDI_VULKAN_SPIRV_INCS "${output_i8}")
    endforeach()
endforeach()
set(gelu_output "${KIDI_VULKAN_GENERATED}/gelu_multiply.inc")
add_custom_command(
    OUTPUT "${gelu_output}"
    COMMAND "${KIDI_VULKAN_GLSLC}" --target-env=vulkan1.3 -O -mfmt=num "${KIDI_VULKAN_GELU_SHADER}"
            -o "${gelu_output}"
    DEPENDS "${KIDI_VULKAN_GELU_SHADER}"
    VERBATIM
)
list(APPEND KIDI_VULKAN_SPIRV_INCS "${gelu_output}")

target_sources(kidi PRIVATE
    src/kidi/tensor/vulkan_backend.cpp
    src/kidi/runtime/vulkan/operators.cpp
    ${KIDI_VULKAN_SPIRV_INCS}
)
target_include_directories(kidi PRIVATE "${KIDI_VULKAN_GENERATED}")
target_compile_definitions(kidi PRIVATE KIDI_HAS_VULKAN=1)
target_link_libraries(kidi PRIVATE vulkan)
