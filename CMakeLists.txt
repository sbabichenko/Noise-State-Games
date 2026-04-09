cmake_minimum_required(VERSION 3.14)
project(NoiseStateGames LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable link-time optimization for cross-TU inlining
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Common compile flags for all targets
add_compile_options(-O3 -march=native -Wall -Wextra)
add_compile_definitions(NDEBUG)

# Eigen3 dependency
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP (optional — graceful fallback if unavailable)
find_package(OpenMP)

add_executable(generate_figures
    generate_figures.cpp
    lqg_solver.cpp
)

target_link_libraries(generate_figures PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(generate_figures PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(solve_interactive
    solve_interactive.cpp
    lqg_solver.cpp
)
target_link_libraries(solve_interactive PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(solve_interactive PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark
    benchmark.cpp
    lqg_solver.cpp
)
target_link_libraries(benchmark PRIVATE Eigen3::Eigen)

add_executable(test_exact_ce
    test_exact_ce.cpp
    lqg_solver.cpp
)
target_link_libraries(test_exact_ce PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_exact_ce PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_ce_picard
    test_ce_picard.cpp
    lqg_solver.cpp
)
target_link_libraries(test_ce_picard PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_ce_picard PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_sweep
    bench_sweep.cpp
    lqg_solver.cpp
)
target_link_libraries(bench_sweep PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_sweep PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_sr_decomp
    test_sr_decomp.cpp
    lqg_solver.cpp
)
target_link_libraries(test_sr_decomp PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_sr_decomp PRIVATE OpenMP::OpenMP_CXX)
endif()
