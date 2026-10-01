if(EMSCRIPTEN)
    if(KIDI_WASM_LARGE_MEMORY)
        string(APPEND CMAKE_C_FLAGS " -sMEMORY64=2")
    endif()
    if(KIDI_WASM_THREADS)
        string(APPEND CMAKE_C_FLAGS " -pthread")
    endif()
endif()

function(kidi_configure_vision_toolchain)
    if(ANDROID)
        target_compile_definitions(pigzpp_lib PRIVATE _LIBCPP_ENABLE_EXPERIMENTAL)
    endif()
    if(CMAKE_CROSSCOMPILING AND CMAKE_TOOLCHAIN_FILE)
        ExternalProject_Add_Step(tahoma_vision_libjpeg kidi_toolchain
            COMMAND "${CMAKE_COMMAND}" --fresh
                -G "${CMAKE_GENERATOR}"
                -S "${CMAKE_CURRENT_SOURCE_DIR}/libs/libjpeg-turbo"
                -B "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-build"
                "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
                "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
                "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                "-DANDROID_ABI=${ANDROID_ABI}"
                "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
            DEPENDEES patch
            DEPENDERS configure
        )
    endif()
endfunction()

cmake_language(DEFER CALL kidi_configure_vision_toolchain)