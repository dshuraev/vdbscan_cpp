cmake_minimum_required(VERSION 3.20)
project(vdbscan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(lib)

option(VDBSCAN_BUILD_CLI "Build the CLI executable" ON)

if(VDBSCAN_BUILD_CLI)
    add_subdirectory(cli)
endif()
