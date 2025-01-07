cmake_minimum_required(VERSION 3.30)
project(rsla)

set(CMAKE_CXX_STANDARD 23)

SET(CMAKE_CXX_FLAGS "-Wall")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -funroll-all-loops")

add_library(rsla_mplapack SHARED rsla.cpp)
add_library(rsla_armadillo SHARED rsla.cpp)

target_compile_definitions(rsla_mplapack PRIVATE "USE_MPLAPACK")
target_compile_definitions(rsla_armadillo PRIVATE "USE_ARMA")

target_include_directories(rsla_mplapack PRIVATE /home/chlorinepentoxide/packages/MPLAPACK/include/mplapack)
target_include_directories(rsla_armadillo PRIVATE /home/chlorinepentoxide/packages/MPLAPACK/include/mplapack)