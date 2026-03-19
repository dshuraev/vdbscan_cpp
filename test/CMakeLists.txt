include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(vdbscan_tests
    test_vdbscan.cpp
)

target_link_libraries(vdbscan_tests PRIVATE
    vdbscan
    Catch2::Catch2WithMain
)

option(VDBSCAN_COVERAGE "Instrument binaries for gcov/llvm-cov line and branch coverage" OFF)

if(VDBSCAN_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        foreach(tgt vdbscan vdbscan_tests)
            target_compile_options(${tgt} PRIVATE --coverage)
            target_link_options(${tgt} PRIVATE --coverage)
        endforeach()
    else()
        message(WARNING "VDBSCAN_COVERAGE is only supported with Clang or GCC")
    endif()
endif()

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(vdbscan_tests)
