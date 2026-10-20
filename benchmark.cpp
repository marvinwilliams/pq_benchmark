#include "heaps.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>
#include <vector>

template <typename T>
std::vector<T> generate_data(std::size_t n, T min_val, T max_val) {
    std::vector<T> data(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<T> dist(min_val, max_val);
    std::generate(data.begin(), data.end(), [&]() { return dist(rng); });
    return data;
}

template <typename PQ>
static void BM_HeapSort(benchmark::State& state) {
    using T = typename PQ::value_type;
    int n = static_cast<int>(state.range(0));
    auto data = generate_data<T>(static_cast<std::size_t>(n), 0, n);
    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        state.ResumeTiming();

        for (auto const& e : data) {
            pq.push(e);
        }
        for (int i = 0; i < n; ++i) {
            pq.pop();
        }
    }
}

template <typename PQ>
static void BM_HeapSortSmall(benchmark::State& state) {
    using T = typename PQ::value_type;
    int n = static_cast<int>(state.range(0));
    auto data = generate_data<T>(static_cast<std::size_t>(n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        state.ResumeTiming();

        for (auto const& e : data) {
            pq.push(e);
        }
        for (int i = 0; i < n; ++i) {
            pq.pop();
        }
    }
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkMonotonic(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        T last = 0;
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(last + data[static_cast<std::size_t>(2 * i)]);
            pq.push(last + data[static_cast<std::size_t>(2 * i + 1)]);
            last = pq.top();
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            last = pq.top();
            pq.pop();
            pq.push(last + data[static_cast<std::size_t>(i)]);
            pq.pop();
        }
    }
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkMonotonicSmall(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        T last = 0;
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(last + data[static_cast<std::size_t>(2 * i)]);
            pq.push(last + data[static_cast<std::size_t>(2 * i + 1)]);
            last = pq.top();
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            last = pq.top();
            pq.pop();
            pq.push(last + data[static_cast<std::size_t>(i)]);
            pq.pop();
        }
    }
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkRandom(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(data[static_cast<std::size_t>(2 * i)]);
            pq.push(data[static_cast<std::size_t>(2 * i + 1)]);
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            pq.pop();
            pq.push(data[static_cast<std::size_t>(i)]);
            pq.pop();
        }
    }
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkRandomSmall(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(data[static_cast<std::size_t>(2 * i)]);
            pq.push(data[static_cast<std::size_t>(2 * i + 1)]);
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            pq.pop();
            pq.push(data[static_cast<std::size_t>(i)]);
            pq.pop();
        }
    }
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopMonotonic(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data<T>(static_cast<std::size_t>(10 * n), T{0},
                                 static_cast<T>(n) << (8 * (sizeof(T) - 4)));

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        for (int i = 0; i < 2 * n; i += 2) {
            pq.push(data[static_cast<std::size_t>(i)]);
        }
        state.ResumeTiming();

        for (int i = 2 * n; i < 10 * n; ++i) {
            auto last = pq.top();
            pq.pop();
            pq.push(last + data[static_cast<std::size_t>(i)]);
        }
    }
    state.SetItemsProcessed(state.iterations() * 8 * n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopMonotonicSmall(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data(static_cast<std::size_t>(2 * n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        for (int i = 0; i < n; ++i) {
            pq.push(data[static_cast<std::size_t>(i)]);
        }
        state.ResumeTiming();

        for (int i = n; i < 2 * n; ++i) {
            auto last = pq.top();
            pq.pop();
            pq.push(last + data[static_cast<std::size_t>(i)]);
        }
    }
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopRandom(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data(static_cast<std::size_t>(2 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        for (int i = 0; i < n; ++i) {
            pq.push(data[static_cast<std::size_t>(i)]);
        }
        state.ResumeTiming();

        for (int i = n; i < 2 * n; ++i) {
            pq.pop();
            pq.push(data[static_cast<std::size_t>(i)]);
        }
    }
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopRandomSmall(benchmark::State& state) {
    int n = static_cast<int>(state.range(0));
    auto data = generate_data(static_cast<std::size_t>(2 * n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        auto pq = PQ();
        for (int i = 0; i < n; ++i) {
            pq.push(data[static_cast<std::size_t>(i)]);
        }
        state.ResumeTiming();

        for (int i = n; i < 2 * n; ++i) {
            pq.pop();
            pq.push(data[static_cast<std::size_t>(i)]);
        }
    }
}

static constexpr int min_range = 1 << 10;
static constexpr int max_range = 1 << 22;

#define REGISTER(NAME, PQ, T)                   \
    BENCHMARK_TEMPLATE(BM_##NAME, PQ<T>)        \
        ->Range(min_range, max_range)           \
        ->Complexity(benchmark::oNLogN);        \
    BENCHMARK_TEMPLATE(BM_##NAME##Small, PQ<T>) \
        ->Range(min_range, max_range)           \
        ->Complexity(benchmark::oNLogN);

#define RUN_BENCHMARKS(NAME, PQ)     \
    REGISTER(NAME, PQ, std::int32_t) \
    REGISTER(NAME, PQ, std::int64_t)

#define RUN_BENCHMARKS_32(NAME, PQ) REGISTER(NAME, PQ, std::int32_t)

#ifdef __AVX512F__
#define AVX512_HEAPS(X, NAME) X(NAME, quick_heap_avx512)
#else
#define AVX512_HEAPS(X, NAME)
#endif

#ifdef WITH_RUST_QUICKHEAP
#define RUST_HEAPS(X, NAME) X(NAME, rust_quick_heap)
#else
#define RUST_HEAPS(X, NAME)
#endif

#define GENERAL_HEAPS(X, NAME) \
    X(NAME, std_pq)            \
    X(NAME, mq_pq)             \
    X(NAME, boost_4_ary_heap)  \
    X(NAME, boost_8_ary_heap)  \
    X(NAME, boost_16_ary_heap) \
    X(NAME, fib_heap)          \
    X(NAME, pairing_heap)      \
    X(NAME, merge_heap)        \
    X(NAME, s3q_queue)         \
    X(NAME, quick_heap)        \
    X(NAME, quick_heap_rs)     \
    X(NAME, quick_heap_avx2)   \
    AVX512_HEAPS(X, NAME)      \
    RUST_HEAPS(X, NAME)

#define GENERAL_HEAPS_32(X, NAME) X(NAME, merge_heap_avx2)

#define MONOTONE_HEAPS(X, NAME) X(NAME, radix_heap)

#define RUN_GENERAL(NAME)               \
    GENERAL_HEAPS(RUN_BENCHMARKS, NAME) \
    GENERAL_HEAPS_32(RUN_BENCHMARKS_32, NAME)

#define RUN_MONOTONE(NAME) \
    RUN_GENERAL(NAME)      \
    MONOTONE_HEAPS(RUN_BENCHMARKS, NAME)

RUN_MONOTONE(HeapSort)
RUN_MONOTONE(GrowShrinkMonotonic)
RUN_GENERAL(GrowShrinkRandom)
RUN_MONOTONE(PushPopMonotonic)
RUN_GENERAL(PushPopRandom)

BENCHMARK_MAIN();
