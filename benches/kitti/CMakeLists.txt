include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
    GIT_SHALLOW TRUE
)

set(KITTI_DATASET_PATH "" CACHE PATH "Path to the directory containing KITTI .bin scans")
set(MAX_FILES 0 CACHE STRING "Maximum number of KITTI scans to benchmark; 0 means all")
set(STRIDE 1 CACHE STRING "Sample every Nth KITTI scan after alphabetical sorting")

if(NOT KITTI_DATASET_PATH)
    message(FATAL_ERROR "KITTI_DATASET_PATH must be set when VDBSCAN_BENCHMARK_KITTI=ON")
endif()

if(NOT IS_DIRECTORY "${KITTI_DATASET_PATH}")
    message(FATAL_ERROR "KITTI_DATASET_PATH must be a directory: ${KITTI_DATASET_PATH}")
endif()

if(STRIDE LESS 1)
    message(FATAL_ERROR "STRIDE must be >= 1")
endif()

if(MAX_FILES LESS 0)
    message(FATAL_ERROR "MAX_FILES must be >= 0")
endif()

set(EPSILON "" CACHE STRING "Override benchmark epsilon (integer, centimetres); if both EPSILON and MINPTS are set, only that single arg-set is benchmarked")
set(MINPTS "" CACHE STRING "Override benchmark min_pts; if both EPSILON and MINPTS are set, only that single arg-set is benchmarked")

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable google benchmark tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable google benchmark gtest integration" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Allow google benchmark dependency downloads" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(vdbscan_bench_kitti
    main.cpp
)

target_include_directories(vdbscan_bench_kitti PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(vdbscan_bench_kitti PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    vdbscan
)

target_compile_features(vdbscan_bench_kitti PRIVATE cxx_std_17)
target_compile_options(vdbscan_bench_kitti PRIVATE -g)
target_compile_definitions(vdbscan_bench_kitti PRIVATE
    KITTI_DATASET_PATH="${KITTI_DATASET_PATH}"
    KITTI_MAX_FILES=${MAX_FILES}
    KITTI_STRIDE=${STRIDE}
)

if(EPSILON AND MINPTS)
    target_compile_definitions(vdbscan_bench_kitti PRIVATE
        KITTI_EPSILON=${EPSILON}
        KITTI_MINPTS=${MINPTS}
    )
    message(STATUS "KITTI benchmark: single arg-set override (epsilon=${EPSILON}, min_pts=${MINPTS})")
endif()

message(STATUS
    "KITTI benchmark configured from ${KITTI_DATASET_PATH} "
    "(stride=${STRIDE}, max=${MAX_FILES})")
