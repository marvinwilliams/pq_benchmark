cmake_minimum_required(VERSION 3.24)

project(
  pq_benchmark
  VERSION 0.1
  LANGUAGES CXX)

include(FetchContent)

option(BUILD_TESTS "Build tests" ON)
find_package(Boost REQUIRED COMPONENTS system)

if(BUILD_TESTS)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
        SYSTEM EXCLUDE_FROM_ALL FIND_PACKAGE_ARGS 3)
  enable_testing()
  add_subdirectory(tests)
endif()

FetchContent_Declare(
  google-benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
      SYSTEM EXCLUDE_FROM_ALL FIND_PACKAGE_ARGS)

FetchContent_Declare(
  tlx
  URL https://github.com/tlx/tlx/archive/2dd63ab839909f0b43cd02108a1869c4670b2c8e.zip
      SYSTEM
      FIND_PACKAGE_ARGS)

add_executable(pq_benchmark benchmark.cpp)
target_compile_features(pq_benchmark PRIVATE cxx_std_23)
target_compile_options(pq_benchmark PRIVATE ${WARNING_FLAGS})
target_include_directories(pq_benchmark PRIVATE .)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_LIBPFM ON)
FetchContent_MakeAvailable(google-benchmark tlx)
target_link_libraries(pq_benchmark PRIVATE benchmark::benchmark tlx Boost::system)
