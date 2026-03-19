include(FetchContent)

find_package(Boost REQUIRED)

if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND CMAKE_CONFIGURATION_TYPES Release)
    list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
        "Supported build configurations" FORCE)
endif()

FetchContent_Declare(
    libmorton
    GIT_REPOSITORY https://github.com/Forceflow/libmorton.git
    GIT_TAG v0.2.12
    GIT_SHALLOW TRUE
)

set(BUILD_TESTING OFF CACHE BOOL "Disable libmorton tests" FORCE)
FetchContent_MakeAvailable(libmorton)

add_library(vdbscan STATIC
    src/index.cpp
    src/vdbscan.cpp
)

target_include_directories(vdbscan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(vdbscan PRIVATE
    Boost::headers
    libmorton::libmorton
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(vdbscan PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-Rpass=loop-vectorize>
        $<$<CONFIG:Release>:-Rpass-missed=loop-vectorize>
        $<$<CONFIG:Release>:-Rpass-analysis=loop-vectorize>
    )
endif()
