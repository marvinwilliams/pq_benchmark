cmake_minimum_required(VERSION 3.24)

project(
  pq_benchmark
  VERSION 0.1
  LANGUAGES CXX)

include(FetchContent)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(WITH_RUST_QUICKHEAP
       "Link the native Rust quickheap (via its C API) as a competitor" OFF)
set(RUST_QUICKHEAP_DIR
    "${CMAKE_SOURCE_DIR}/../quickheap/quickheap"
    CACHE PATH "Path to the Rust quickheap crate")
set(WARNING_FLAGS
    ""
    CACHE STRING "Warning flags for this project's targets (set by the presets)")

add_library(pq_warnings INTERFACE)
target_compile_options(pq_warnings INTERFACE ${WARNING_FLAGS})

if(WITH_RUST_QUICKHEAP)
  find_program(CARGO cargo REQUIRED)
  find_package(Threads REQUIRED)
  set(RUST_QUICKHEAP_TARGET_DIR "${CMAKE_BINARY_DIR}/rust-quickheap")
  set(RUST_QUICKHEAP_LIB "${RUST_QUICKHEAP_TARGET_DIR}/release/libquickheap.a")
  # Uses the crate's .cargo/config.toml (target-cpu=native). Cargo decides
  # itself whether anything needs rebuilding.
  add_custom_target(
    rust_quickheap_build
    COMMAND ${CARGO} rustc --release --lib --features c --crate-type staticlib
            --target-dir "${RUST_QUICKHEAP_TARGET_DIR}"
    WORKING_DIRECTORY "${RUST_QUICKHEAP_DIR}"
    BYPRODUCTS "${RUST_QUICKHEAP_LIB}"
    USES_TERMINAL)
  add_library(rust_quickheap STATIC IMPORTED GLOBAL)
  set_target_properties(
    rust_quickheap
    PROPERTIES IMPORTED_LOCATION "${RUST_QUICKHEAP_LIB}"
               INTERFACE_INCLUDE_DIRECTORIES "${RUST_QUICKHEAP_DIR}/cpp"
               INTERFACE_COMPILE_DEFINITIONS WITH_RUST_QUICKHEAP
               INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m")
  add_dependencies(rust_quickheap rust_quickheap_build)
endif()

if(BUILD_TESTS)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
        SYSTEM EXCLUDE_FROM_ALL FIND_PACKAGE_ARGS 3)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  FetchContent_Declare(
    google-benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
        SYSTEM EXCLUDE_FROM_ALL)

  FetchContent_Declare(
    tlx
    URL https://github.com/tlx/tlx/archive/2dd63ab839909f0b43cd02108a1869c4670b2c8e.zip
        SYSTEM)

  find_package(Boost)
  # Before s3q, which would otherwise populate its own tlx without adding it.
  FetchContent_MakeAvailable(tlx)
  add_subdirectory(competitors/s3q)
  add_executable(pq_benchmark benchmark.cpp)
  target_compile_features(pq_benchmark PRIVATE cxx_std_23)
  target_include_directories(pq_benchmark PRIVATE .)

  set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
  set(BENCHMARK_ENABLE_LIBPFM ON)
  FetchContent_MakeAvailable(google-benchmark)
  target_link_libraries(pq_benchmark PRIVATE pq_warnings benchmark::benchmark tlx
                                             Boost::boost s3q)
  if(WITH_RUST_QUICKHEAP)
    target_link_libraries(pq_benchmark PRIVATE rust_quickheap)
  endif()
endif()
