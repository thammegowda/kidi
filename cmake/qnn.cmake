include_guard(GLOBAL)

if(KIDI_QNN_SDK AND EXISTS "${KIDI_QNN_SDK}/include/QNN/QnnInterface.h")
    target_sources(kidi PRIVATE src/kidi/runtime/qnn/compiler.cpp)
    target_include_directories(kidi SYSTEM PRIVATE "${KIDI_QNN_SDK}/include/QNN")
    target_compile_definitions(kidi PRIVATE KIDI_HAS_QNN=1)
    target_link_libraries(kidi PRIVATE ${CMAKE_DL_LIBS})
endif()
