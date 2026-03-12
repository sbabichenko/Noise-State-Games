cmake_minimum_required(VERSION 3.14)
project(NoiseStateGames LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 dependency
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(generate_figures
    generate_figures.cpp
    lqg_solver.cpp
)

target_link_libraries(generate_figures PRIVATE Eigen3::Eigen)
target_compile_options(generate_figures PRIVATE -O2 -Wall -Wextra)
