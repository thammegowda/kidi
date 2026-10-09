if(EMSCRIPTEN)
    string(APPEND CMAKE_C_FLAGS " ${KIDI_WASM_ADDRESS_FLAG}")
    if(KIDI_WASM_THREADS)
        string(APPEND CMAKE_C_FLAGS " -pthread")
    endif()
endif()

function(kidi_configure_vision_toolchain)
    if(ANDROID)
        target_compile_definitions(pigzpp_lib PRIVATE _LIBCPP_ENABLE_EXPERIMENTAL)
    endif()
endfunction()

cmake_language(DEFER CALL kidi_configure_vision_toolchain)