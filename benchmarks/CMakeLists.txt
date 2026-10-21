add_library(benchmark_base INTERFACE)
target_compile_features(benchmark_base INTERFACE cxx_std_23)
target_include_directories(benchmark_base INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_base INTERFACE pq_warnings)

if(PAPI_FOUND)
  target_link_libraries(benchmark_base INTERFACE PAPI::PAPI)
  target_compile_definitions(benchmark_base INTERFACE WITH_PAPI)
else()
  message(STATUS "PAPI not found, performance counters not available")
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#ifndef __AVX512F__
#error
#endif
int main() {}" HAVE_AVX512F)

set(PQS
    std_pq
    mq_pq
    boost_4_ary_heap
    boost_8_ary_heap
    boost_16_ary_heap
    fib_heap
    pairing_heap
    merge_heap
    merge_heap_avx2
    s3q_queue
    radix_heap
    quick_heap
    quick_heap_rs
    quick_heap_avx2
    stronger_quickheap)
if(HAVE_AVX512F)
  list(APPEND PQS quick_heap_avx512)
endif()
if(WITH_RUST_QUICKHEAP)
  list(APPEND PQS rust_quick_heap)
endif()

set(PQ_HEADER_boost_4_ary_heap boost_d_ary_heap)
set(PQ_HEADER_boost_8_ary_heap boost_d_ary_heap)
set(PQ_HEADER_boost_16_ary_heap boost_d_ary_heap)
set(PQ_HEADER_quick_heap_avx2 quick_heap_simd)
set(PQ_HEADER_quick_heap_avx512 quick_heap_simd)

set(PQ_LIBS_boost_d_ary_heap Boost::boost)
set(PQ_LIBS_fib_heap Boost::boost)
set(PQ_LIBS_pairing_heap Boost::boost)
set(PQ_LIBS_radix_heap tlx)
set(PQ_LIBS_s3q_queue s3q)
set(PQ_LIBS_rust_quick_heap rust_quickheap)

function(add_pq_benchmarks name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "SOURCE" "LIBS;PQS")
  add_custom_target(${name})
  foreach(pq IN LISTS ARG_PQS)
    if(DEFINED PQ_HEADER_${pq})
      set(header ${PQ_HEADER_${pq}})
    else()
      set(header ${pq})
    endif()
    add_executable(${name}_${pq} ${ARG_SOURCE})
    target_compile_definitions(${name}_${pq} PRIVATE BENCH_PQ=${pq}
                                                     BENCH_PQ_HEADER="benchmarks/heaps/${header}.hpp")
    target_link_libraries(${name}_${pq} PRIVATE benchmark_base ${ARG_LIBS} ${PQ_LIBS_${header}})
    add_dependencies(${name} ${name}_${pq})
  endforeach()
endfunction()

add_pq_benchmarks(workloads SOURCE workloads.cpp LIBS benchmark::benchmark PQS ${PQS})
add_pq_benchmarks(dijkstra SOURCE dijkstra.cpp PQS ${PQS})
add_pq_benchmarks(knapsack SOURCE knapsack.cpp PQS ${PQS})
set(PRIM_PQS ${PQS})
list(REMOVE_ITEM PRIM_PQS radix_heap)
add_pq_benchmarks(prim SOURCE prim.cpp PQS ${PRIM_PQS})
