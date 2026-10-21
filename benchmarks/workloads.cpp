#include "pq.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

template <typename T>
class Keys {
    std::mt19937 rng_{42};
    std::uniform_int_distribution<T> dist_;
    std::vector<T> keys_;

   public:
    Keys(std::size_t n, T min_val, T max_val)
        : dist_(min_val, max_val), keys_(n) {}

    std::vector<T> const& next() {
        std::ranges::generate(keys_, [&] { return dist_(rng_); });
        return keys_;
    }
};

template <typename T>
using Queue = std::optional<BENCH_PQ<T>>;

template <typename T>
static void heap_sort(benchmark::State& state, T max_key) {
    auto n = static_cast<std::size_t>(state.range(0));
    Keys<T> keys(n, 0, max_key);
    Queue<T> queue;
    for (auto _ : state) {
        state.PauseTiming();
        auto const& data = keys.next();
        auto& pq = queue.emplace();
        state.ResumeTiming();

        for (auto const& e : data) {
            pq.push(e);
        }
        for (std::size_t i = 0; i < n; ++i) {
            pq.pop();
        }
    }
    state.SetItemsProcessed(state.iterations() * 2 * state.range(0));
}

template <typename T>
static void grow_shrink_monotonic(benchmark::State& state, T max_key) {
    auto n = static_cast<std::size_t>(state.range(0));
    Keys<T> keys(3 * n, 0, max_key);
    Queue<T> queue;
    for (auto _ : state) {
        state.PauseTiming();
        auto const& data = keys.next();
        auto& pq = queue.emplace();
        T last = 0;
        state.ResumeTiming();
        std::size_t i = 0;
        for (; i < 2 * n; i += 2) {
            pq.push(last + data[i]);
            pq.push(last + data[i + 1]);
            last = pq.top();
            pq.pop();
        }
        for (; i < 3 * n; ++i) {
            last = pq.top();
            pq.pop();
            pq.push(last + data[i]);
            pq.pop();
        }
    }
    state.SetItemsProcessed(state.iterations() * 6 * state.range(0));
}

template <typename T>
static void grow_shrink_random(benchmark::State& state, T max_key) {
    auto n = static_cast<std::size_t>(state.range(0));
    Keys<T> keys(3 * n, 0, max_key);
    Queue<T> queue;
    for (auto _ : state) {
        state.PauseTiming();
        auto const& data = keys.next();
        auto& pq = queue.emplace();
        state.ResumeTiming();
        std::size_t i = 0;
        for (; i < 2 * n; i += 2) {
            pq.push(data[i]);
            pq.push(data[i + 1]);
            pq.pop();
        }
        for (; i < 3 * n; ++i) {
            pq.pop();
            pq.push(data[i]);
            pq.pop();
        }
    }
    state.SetItemsProcessed(state.iterations() * 6 * state.range(0));
}

template <typename T>
static void push_pop_monotonic(benchmark::State& state, T max_key,
                               std::size_t rounds) {
    auto n = static_cast<std::size_t>(state.range(0));
    Keys<T> keys((1 + rounds) * n, 0, max_key);
    Queue<T> queue;
    for (auto _ : state) {
        state.PauseTiming();
        auto const& data = keys.next();
        auto& pq = queue.emplace();
        for (std::size_t i = 0; i < n; ++i) {
            pq.push(data[i]);
        }
        state.ResumeTiming();

        for (std::size_t i = n; i < (1 + rounds) * n; ++i) {
            auto last = pq.top();
            pq.pop();
            pq.push(last + data[i]);
        }
    }
    state.SetItemsProcessed(state.iterations() *
                            static_cast<std::int64_t>(2 * rounds * n));
}

template <typename T>
static void BM_HeapSort(benchmark::State& state) {
    heap_sort<T>(state, static_cast<T>(state.range(0)));
}

template <typename T>
static void BM_HeapSortSmall(benchmark::State& state) {
    heap_sort<T>(state, 1000);
}

template <typename T>
static void BM_GrowShrinkMonotonic(benchmark::State& state) {
    grow_shrink_monotonic<T>(state, static_cast<T>(state.range(0)));
}

template <typename T>
static void BM_GrowShrinkMonotonicSmall(benchmark::State& state) {
    grow_shrink_monotonic<T>(state, 1000);
}

template <typename T>
static void BM_GrowShrinkRandom(benchmark::State& state) {
    grow_shrink_random<T>(state, static_cast<T>(state.range(0)));
}

template <typename T>
static void BM_GrowShrinkRandomSmall(benchmark::State& state) {
    grow_shrink_random<T>(state, 1000);
}

template <typename T>
static void BM_PushPopMonotonic(benchmark::State& state) {
    push_pop_monotonic<T>(
        state,
        static_cast<T>(static_cast<T>(state.range(0)) << (8 * (sizeof(T) - 4))),
        8);
}

template <typename T>
static void BM_PushPopMonotonicSmall(benchmark::State& state) {
    push_pop_monotonic<T>(state, 1000, 1);
}

static constexpr int min_range = 1 << 10;
static constexpr int max_range = 1 << 25;

#define REGISTER(NAME, T)                   \
    BENCHMARK_TEMPLATE(BM_##NAME, T)        \
        ->RangeMultiplier(2)                \
        ->Range(min_range, max_range);      \
    BENCHMARK_TEMPLATE(BM_##NAME##Small, T) \
        ->RangeMultiplier(2)                \
        ->Range(min_range, max_range);

#define RUN(NAME)                \
    REGISTER(NAME, std::int32_t) \
    REGISTER(NAME, std::int64_t)

RUN(HeapSort)
RUN(GrowShrinkMonotonic)
#ifndef PQ_MONOTONE
RUN(GrowShrinkRandom)
#endif
RUN(PushPopMonotonic)

BENCHMARK_MAIN();
