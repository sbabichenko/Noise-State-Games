cmake_minimum_required(VERSION 3.14)
project(NoiseStateGames LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 dependency
find_package(Eigen3 REQUIRED NO_MODULE)

# OpenMP (optional — graceful fallback if unavailable)
find_package(OpenMP)

add_executable(generate_figures
    generate_figures.cpp
    lqg_solver.cpp
)

target_link_libraries(generate_figures PRIVATE Eigen3::Eigen)
target_compile_options(generate_figures PRIVATE -O3 -march=native -Wall -Wextra)

if(OpenMP_CXX_FOUND)
    target_link_libraries(generate_figures PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(solve_interactive
    solve_interactive.cpp
    lqg_solver.cpp
)
target_link_libraries(solve_interactive PRIVATE Eigen3::Eigen)
target_compile_options(solve_interactive PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(solve_interactive PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark
    benchmark.cpp
    lqg_solver.cpp
)
target_link_libraries(benchmark PRIVATE Eigen3::Eigen)
target_compile_options(benchmark PRIVATE -O3 -march=native -Wall -Wextra)

add_executable(test_exact_ce
    test_exact_ce.cpp
    lqg_solver.cpp
)
target_link_libraries(test_exact_ce PRIVATE Eigen3::Eigen)
target_compile_options(test_exact_ce PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_exact_ce PRIVATE OpenMP::OpenMP_CXX)
endif()
