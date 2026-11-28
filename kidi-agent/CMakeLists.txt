cmake_minimum_required(VERSION 3.25)

set(KIDI_AGENT_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(KIDI_AGENT_STANDALONE ON)
    project(kidi_agent VERSION 0.1.0 LANGUAGES C CXX)
    include(CTest)
endif()

option(KIDI_AGENT_BUILD_CLI "Build the standalone kidi-agent executable" ${KIDI_AGENT_STANDALONE})
get_filename_component(KIDI_AGENT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(NOT TARGET nlohmann_json::nlohmann_json)
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    set(JSON_Install OFF CACHE BOOL "" FORCE)
    add_subdirectory("${KIDI_AGENT_ROOT}/third_party/nlohmann-json" "${CMAKE_CURRENT_BINARY_DIR}/json" EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${KIDI_AGENT_ROOT}/third_party/yaml-cpp" "${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp" EXCLUDE_FROM_ALL)
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/agent.yaml")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/agent.yaml" KIDI_AGENT_YAML)
configure_file(src/kidi/agent/default_config.h.in generated/default_config.h @ONLY)

# The workspace tools are built on LiteApp's root-confined workspace (also used by the web UI's file services).
if(NOT TARGET liteapp::workspace)
    set(LITEAPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${KIDI_AGENT_ROOT}/third_party/liteapp" "${CMAKE_CURRENT_BINARY_DIR}/liteapp" EXCLUDE_FROM_ALL)
endif()

add_library(kidi_agent
    src/kidi/agent/agent.cpp
    src/kidi/agent/config.cpp
    src/kidi/agent/chat_client.cpp
    src/kidi/agent/workspace.cpp
)
add_library(kidi::agent ALIAS kidi_agent)
target_compile_features(kidi_agent PUBLIC cxx_std_23)
target_include_directories(kidi_agent PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" "${KIDI_AGENT_ROOT}/src")
target_include_directories(kidi_agent PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(kidi_agent PUBLIC nlohmann_json::nlohmann_json PRIVATE yaml-cpp::yaml-cpp liteapp::workspace)

add_library(kidi_agent_http src/kidi/agent/chat_http.cpp)
add_library(kidi::agent_http ALIAS kidi_agent_http)
target_link_libraries(kidi_agent_http PUBLIC kidi::agent PRIVATE liteapp::httplib)

if(TARGET kidi_cli_main OR KIDI_AGENT_BUILD_CLI)
    if(NOT TARGET kidi::argparse)
        add_library(kidi_agent_argparse "${KIDI_AGENT_ROOT}/src/kidi/cli/argparse.cpp")
        add_library(kidi::argparse ALIAS kidi_agent_argparse)
        target_compile_features(kidi_agent_argparse PUBLIC cxx_std_23)
        target_include_directories(kidi_agent_argparse PUBLIC "${KIDI_AGENT_ROOT}/src")
    endif()
    add_library(kidi_agent_command src/kidi/agent/command.cpp src/kidi/agent/markdown.cpp src/kidi/agent/session.cpp
        src/kidi/agent/terminal.cpp src/kidi/agent/transcript.cpp)
    target_link_libraries(kidi_agent_command PUBLIC kidi::agent kidi::argparse)
    if(UNIX)
        find_package(Threads REQUIRED)
        target_link_libraries(kidi_agent_command PRIVATE Threads::Threads)
    endif()
    # `kidi agent --web` embeds the files web/assets.txt lists; --web-assets serves the same manifest from a checkout.
    if(UNIX AND TARGET liteapp::app AND TARGET liteapp::service AND
       COMMAND liteapp_embed_manifest)
        liteapp_embed_manifest(
            TARGET kidi_agent_web_assets
            NAMESPACE kidi::agent_web_assets
            HEADER kidi/agent_web_assets.h
            MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/web/assets.txt"
            BASE "${KIDI_AGENT_ROOT}"
        )
        target_sources(kidi_agent_command PRIVATE src/kidi/agent/web.cpp)
        target_link_libraries(kidi_agent_command PRIVATE kidi_agent_web_assets liteapp::app liteapp::service
            liteapp::vfs liteapp::workspace)
        target_compile_definitions(kidi_agent_command PRIVATE KIDI_AGENT_WEB=1)
    endif()
    if(KIDI_AGENT_BUILD_CLI)
        add_executable(kidi_agent_cli src/main.cpp)
        set_target_properties(kidi_agent_cli PROPERTIES OUTPUT_NAME kidi-agent)
        target_link_libraries(kidi_agent_cli PRIVATE kidi_agent_command kidi::agent_http)
    endif()
endif()

if(KIDI_AGENT_STANDALONE AND BUILD_TESTING)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    add_subdirectory("${KIDI_AGENT_ROOT}/third_party/ynnpack-dev/third_party/googletest"
        "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)
    include(GoogleTest)
    add_executable(kidi_agent_test tests/agent_test.cpp tests/config_test.cpp tests/workspace_test.cpp)
    target_link_libraries(kidi_agent_test PRIVATE kidi::agent GTest::gtest_main)
    gtest_discover_tests(kidi_agent_test)
    add_executable(kidi_agent_chat_client_test tests/chat_client_test.cpp)
    target_link_libraries(kidi_agent_chat_client_test PRIVATE kidi::agent_http liteapp::httplib GTest::gtest_main)
    gtest_discover_tests(kidi_agent_chat_client_test)
    if(TARGET kidi_agent_command AND UNIX)
        add_executable(kidi_agent_command_test tests/command_test.cpp)
        target_link_libraries(kidi_agent_command_test PRIVATE kidi_agent_command GTest::gtest_main)
        gtest_discover_tests(kidi_agent_command_test PROPERTIES TIMEOUT 20)
    endif()
endif()
