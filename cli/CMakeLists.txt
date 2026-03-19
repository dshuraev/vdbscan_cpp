include(FetchContent)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(CLI11)

add_executable(vdbscan_cli
    codecs.cpp
    main.cpp
)

target_include_directories(vdbscan_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vdbscan_cli PRIVATE
    CLI11::CLI11
    vdbscan
)
