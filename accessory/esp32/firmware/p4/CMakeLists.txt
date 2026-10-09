cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(EXTRA_COMPONENT_DIRS "../components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(kidi_accessory_p4)
