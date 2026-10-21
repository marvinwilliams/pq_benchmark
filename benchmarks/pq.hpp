#pragma once

#if !defined(BENCH_PQ) || !defined(BENCH_PQ_HEADER)
#error "BENCH_PQ and BENCH_PQ_HEADER must be defined"
#endif

#include BENCH_PQ_HEADER

#include <string_view>

#define BENCH_PQ_STRINGIFY_IMPL(x) #x
#define BENCH_PQ_STRINGIFY(x) BENCH_PQ_STRINGIFY_IMPL(x)

inline constexpr std::string_view pq_name = BENCH_PQ_STRINGIFY(BENCH_PQ);
