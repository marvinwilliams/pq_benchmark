#include "competitors/merge_heap.hpp"
#include "competitors/mq_heap.hpp"
#include "competitors/quick_heap.hpp"
#include "competitors/quick_heap_avx2.hpp"

#include <tlx/container/radix_heap.hpp>

#include <benchmark/benchmark.h>

#include <boost/heap/d_ary_heap.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/heap/pairing_heap.hpp>

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <vector>

template <typename T>
using std_pq = std::priority_queue<T, std::vector<T>, std::ranges::greater>;

template <typename T>
using mq_pq = multiqueue::Heap<T, std::ranges::greater>;

template <typename T>
using merge_heap = multiqueue::value_merge_heap<T, std::ranges::less>;

template <typename T>
using boost_4_ary_heap =
    boost::heap::d_ary_heap<T, boost::heap::arity<4>,
                            boost::heap::compare<std::ranges::greater>>;
template <typename T>
using boost_8_ary_heap =
    boost::heap::d_ary_heap<T, boost::heap::arity<8>,
                            boost::heap::compare<std::ranges::greater>>;

template <typename T>
using boost_16_ary_heap =
    boost::heap::d_ary_heap<T, boost::heap::arity<16>,
                            boost::heap::compare<std::ranges::greater>>;

template <typename T>
using fib_heap =
    boost::heap::fibonacci_heap<T, boost::heap::compare<std::ranges::greater>>;

template <typename T>
using pairing_heap =
    boost::heap::pairing_heap<T, boost::heap::compare<std::ranges::greater>>;

template <typename T>
using radix_heap = tlx::RadixHeap<T, std::identity, T>;

template <typename T>
using quick_heap = QuickHeap<T>;

template <typename T>
using quick_heap_avx2 = QuickHeapAVX2<T>;

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
    int n = state.range(0);
    auto data = generate_data<T>(static_cast<std::size_t>(n), 0, n);
    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        state.ResumeTiming();

        for (auto const& e : data) {
            pq.push(e);
        }
        for (int i = 0; i < n; ++i) {
            pq.pop();
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ>
static void BM_HeapSortSmall(benchmark::State& state) {
    using T = typename PQ::value_type;
    int n = state.range(0);
    auto data = generate_data<T>(static_cast<std::size_t>(n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        state.ResumeTiming();

        for (auto const& e : data) {
            pq.push(e);
        }
        for (int i = 0; i < n; ++i) {
            pq.pop();
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkMonotonic(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        T last = 0;
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(last + data[2 * i]);
            pq.push(last + data[2 * i + 1]);
            last = pq.top();
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            last = pq.top();
            pq.pop();
            pq.push(last + data[i]);
            pq.pop();
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkMonotonicSmall(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        T last = 0;
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(last + data[2 * i]);
            pq.push(last + data[2 * i + 1]);
            last = pq.top();
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            last = pq.top();
            pq.pop();
            pq.push(last + data[i]);
            pq.pop();
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkRandom(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        T last = 0;
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(data[2 * i]);
            pq.push(data[2 * i + 1]);
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            pq.pop();
            pq.push(data[i]);
            pq.pop();
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_GrowShrinkRandomSmall(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(3 * n), 0, 1000);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        T last = 0;
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) {
            pq.push(data[2 * i]);
            pq.push(data[2 * i + 1]);
            pq.pop();
        }
        for (int i = 2 * n; i < 3 * n; ++i) {
            pq.pop();
            pq.push(data[i]);
            pq.pop();
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopMonotonic(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(2 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        for (int i = 0; i < n; ++i) {
            pq.push(data[i]);
        }
        state.ResumeTiming();

        for (int i = n; i < 2 * n; ++i) {
            auto last = pq.top();
            pq.pop();
            pq.push(last + data[i]);
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopMonotonicSmall(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(2 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        for (int i = 0; i < n; ++i) {
            pq.push(data[i]);
        }
        state.ResumeTiming();

        for (int i = n; i < 2 * n; ++i) {
            auto last = pq.top();
            pq.pop();
            pq.push(last + data[i]);
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopRandom(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(2 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        for (int i = 0; i < n; ++i) {
            pq.push(data[i]);
        }
        state.ResumeTiming();

        for (int i = n; i < 2 * n; ++i) {
            pq.pop();
            pq.push(data[i]);
        }
    }
    state.SetComplexityN(n);
}

template <typename PQ, typename T = typename PQ::value_type>
static void BM_PushPopRandomSmall(benchmark::State& state) {
    int n = state.range(0);
    auto data = generate_data(static_cast<std::size_t>(2 * n), 0, n);

    for (auto _ : state) {
        state.PauseTiming();
        PQ pq;
        for (int i = 0; i < n; ++i) {
            pq.push(data[i]);
        }
        state.ResumeTiming();

        for (int i = n; i < 2 * n; ++i) {
            pq.pop();
            pq.push(data[i]);
        }
    }
    state.SetComplexityN(n);
}

static constexpr int min_range = 1 << 10;
static constexpr int max_range = 1 << 22;

#define RUN_BENCHMARKS(NAME, PQ)                           \
    BENCHMARK_TEMPLATE(BM_##NAME, PQ<std::int32_t>)        \
        ->Range(min_range, max_range)                      \
        ->Complexity(benchmark::oNLogN);                   \
    BENCHMARK_TEMPLATE(BM_##NAME##Small, PQ<std::int32_t>) \
        ->Range(min_range, max_range)                      \
        ->Complexity(benchmark::oNLogN);                   
    // BENCHMARK_TEMPLATE(BM_##NAME, PQ<std::int64_t>)        \
    //     ->Range(min_range, max_range)                      \
    //     ->Complexity(benchmark::oNLogN);                   \
    // BENCHMARK_TEMPLATE(BM_##NAME##Small, PQ<std::int64_t>) \
    //     ->Range(min_range, max_range)                      \
    //     ->Complexity(benchmark::oNLogN);

RUN_BENCHMARKS(HeapSort, std_pq)
// RUN_BENCHMARKS(HeapSort, mq_pq)
RUN_BENCHMARKS(HeapSort, merge_heap)
RUN_BENCHMARKS(HeapSort, boost_4_ary_heap)
RUN_BENCHMARKS(HeapSort, radix_heap)
RUN_BENCHMARKS(HeapSort, quick_heap)
RUN_BENCHMARKS(HeapSort, quick_heap_avx2)

RUN_BENCHMARKS(GrowShrinkMonotonic, std_pq)
// RUN_BENCHMARKS(GrowShrinkMonotonic, mq_pq)
RUN_BENCHMARKS(GrowShrinkMonotonic, merge_heap)
RUN_BENCHMARKS(GrowShrinkMonotonic, boost_4_ary_heap)
RUN_BENCHMARKS(GrowShrinkMonotonic, radix_heap)
RUN_BENCHMARKS(GrowShrinkMonotonic, quick_heap)
RUN_BENCHMARKS(GrowShrinkMonotonic, quick_heap_avx2)

RUN_BENCHMARKS(GrowShrinkRandom, std_pq)
// RUN_BENCHMARKS(GrowShrinkRandom, mq_pq)
RUN_BENCHMARKS(GrowShrinkRandom, merge_heap)
RUN_BENCHMARKS(GrowShrinkRandom, boost_4_ary_heap)
RUN_BENCHMARKS(GrowShrinkRandom, quick_heap)
RUN_BENCHMARKS(GrowShrinkRandom, quick_heap_avx2)

RUN_BENCHMARKS(PushPopMonotonic, std_pq)
// RUN_BENCHMARKS(PushPopMonotonic, mq_pq)
RUN_BENCHMARKS(PushPopMonotonic, merge_heap)
RUN_BENCHMARKS(PushPopMonotonic, boost_4_ary_heap)
RUN_BENCHMARKS(PushPopMonotonic, radix_heap)
RUN_BENCHMARKS(PushPopMonotonic, quick_heap)
RUN_BENCHMARKS(PushPopMonotonic, quick_heap_avx2)

RUN_BENCHMARKS(PushPopRandom, std_pq)
// RUN_BENCHMARKS(PushPopRandom, mq_pq)
RUN_BENCHMARKS(PushPopRandom, merge_heap)
RUN_BENCHMARKS(PushPopRandom, boost_4_ary_heap)
RUN_BENCHMARKS(PushPopRandom, quick_heap)
RUN_BENCHMARKS(PushPopRandom, quick_heap_avx2)

BENCHMARK_MAIN();
