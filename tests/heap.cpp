#include "competitors/merge_heap.hpp"
#include "competitors/merge_heap_simd.hpp"
#include "competitors/mq_heap.hpp"
#include "competitors/quick_heap_scalar.hpp"
#include "competitors/quick_heap_simd.hpp"
#include "competitors/quick_heap_rust_port.hpp"

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"

#ifdef __AVX512F__
#define AVX512_HEAPS                                                     \
    , (QuickHeapAVX512<std::int32_t>), (QuickHeapAVX512<std::int64_t>), \
        (QuickHeapAVX512<std::int32_t, true>),                           \
        (QuickHeapAVX512<std::int64_t, true>),                           \
        (QuickHeapAVX512<std::int32_t, false, false>),                   \
        (QuickHeapAVX512<std::int64_t, false, false>),                   \
        (QuickHeapAVX512<std::int32_t, false, true, Scan::simd>),        \
        (QuickHeapAVX512<std::int64_t, false, true, Scan::simd>)
#else
#define AVX512_HEAPS
#endif

#ifdef WITH_RUST_QUICKHEAP
#include "competitors/quick_heap_rust_wrapper.hpp"
#define RUST_HEAPS , (RustQuickHeap<std::int32_t>), (RustQuickHeap<std::int64_t>)
#else
#define RUST_HEAPS
#endif

#include <array>
#include <limits>
#include <list>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

TEMPLATE_TEST_CASE("heap supports basic operations", "[heap][basic]",
                   (multiqueue::Heap<int, std::ranges::greater>),
                   (QuickHeap<int, std::ranges::less>),
                   (QuickHeapAVX2<std::int32_t>),
                   (QuickHeapAVX2<std::int64_t>),
                   (QuickHeapAVX2<std::int32_t, true>),
                   (QuickHeapAVX2<std::int64_t, true>),
                   (QuickHeapAVX2<std::int32_t, false, false>),
                   (QuickHeapAVX2<std::int64_t, false, false>),
                   (QuickHeapAVX2<std::int32_t, false, true, Scan::simd>),
                   (QuickHeapAVX2<std::int64_t, false, true, Scan::simd>),
                   (QuickHeapRs<std::int32_t>),
                   (QuickHeapRs<std::int64_t>),
                   (multiqueue::value_merge_heap<int, std::ranges::less>),
                   (MergeHeapAVX2<std::int32_t>),
                   (MergeHeapAVX2<std::int64_t>) AVX512_HEAPS RUST_HEAPS) {
    using heap_t = TestType;

    auto heap = heap_t{};

    SECTION("push increasing numbers and pop them") {
        for (int n = 0; n < 1000; ++n) {
            heap.push(n);
        }

        for (int i = 0; i < 1000; ++i) {
            REQUIRE(heap.top() == i);
            heap.pop();
        }
        REQUIRE(heap.empty());
    }

    SECTION("push decreasing numbers and pop them") {
        for (int n = 1000; n > 0; --n) {
            heap.push(n - 1);
        }

        for (int i = 0; i < 1000; ++i) {
            REQUIRE(heap.top() == i);
            heap.pop();
        }
        REQUIRE(heap.empty());
    }

    SECTION(
        "first push increasing numbers, then push decreasing numbers and "
        "pop them") {
        for (int i = 0; i < 500; ++i) {
            heap.push(i);
        }
        for (int i = 1000; i > 500; --i) {
            heap.push(i - 1);
        }
        for (int i = 0; i < 1000; ++i) {
            REQUIRE(heap.top() == i);
            heap.pop();
        }
        REQUIRE(heap.empty());
    }
}

TEMPLATE_TEST_CASE("max heap", "[heap][comparator]",
                   (multiqueue::Heap<int, std::ranges::less>),
                   (QuickHeap<int, std::ranges::greater>),
                   (multiqueue::value_merge_heap<int, std::ranges::greater>)) {
    using heap_t = TestType;

    auto heap = heap_t{};

    SECTION("push increasing numbers and pop them") {
        for (int n = 0; n < 1000; ++n) {
            heap.push(n);
        }

        for (int i = 0; i < 1000; ++i) {
            REQUIRE(heap.top() == 999 - i);
            heap.pop();
        }
        REQUIRE(heap.empty());
    }

    SECTION("push decreasing numbers and pop them") {
        for (int n = 999; n >= 0; --n) {
            heap.push(n);
        }

        for (int i = 0; i < 1000; ++i) {
            REQUIRE(heap.top() == 999 - i);
            heap.pop();
        }
        REQUIRE(heap.empty());
    }

    SECTION(
        "first push increasing numbers, then push decreasing numbers and "
        "pop them") {
        for (int i = 0; i < 500; ++i) {
            heap.push(i);
        }
        for (int i = 999; i >= 500; --i) {
            heap.push(i);
        }
        for (int i = 0; i < 1000; ++i) {
            REQUIRE(heap.top() == 999 - i);
            heap.pop();
        }
        REQUIRE(heap.empty());
    }
}

TEMPLATE_TEST_CASE("heap works with randomized workloads", "[heap][workloads]",
                   (multiqueue::Heap<int, std::ranges::greater>),
                   (QuickHeap<int, std::ranges::less>),
                   (QuickHeapAVX2<std::int32_t>),
                   (QuickHeapAVX2<std::int64_t>),
                   (QuickHeapAVX2<std::int32_t, true>),
                   (QuickHeapAVX2<std::int64_t, true>),
                   (QuickHeapAVX2<std::int32_t, false, false>),
                   (QuickHeapAVX2<std::int64_t, false, false>),
                   (QuickHeapAVX2<std::int32_t, false, true, Scan::simd>),
                   (QuickHeapAVX2<std::int64_t, false, true, Scan::simd>),
                   (QuickHeapRs<std::int32_t>),
                   (QuickHeapRs<std::int64_t>),
                   (multiqueue::value_merge_heap<int, std::ranges::less>),
                   (MergeHeapAVX2<std::int32_t>),
                   (MergeHeapAVX2<std::int64_t>) AVX512_HEAPS RUST_HEAPS) {
    using heap_t = TestType;

    auto heap = heap_t{};
    auto ref_pq = std::priority_queue<int, std::vector<int>, std::greater<>>{};
    auto gen = std::mt19937{0};

    SECTION("push random numbers and pop them") {
        auto dist = std::uniform_int_distribution{-100, 100};

        for (std::size_t i = 0; i < 1000; ++i) {
            auto n = dist(gen);
            heap.push(n);
            ref_pq.push(n);
            REQUIRE(heap.top() == ref_pq.top());
        }

        for (std::size_t i = 0; i < 1000; ++i) {
            REQUIRE(heap.top() == ref_pq.top());
            heap.pop();
            ref_pq.pop();
        }
        REQUIRE(heap.empty());
        REQUIRE(ref_pq.empty());
    }

    SECTION("interleave pushing and popping random numbers") {
        auto dist = std::uniform_int_distribution{-100, 100};
        auto seq_dist = std::uniform_int_distribution{0, 10};

        for (int s = 0; s < 1000; ++s) {
            auto num_push = seq_dist(gen);
            for (int i = 0; i < num_push; ++i) {
                auto n = dist(gen);
                heap.push(n);
                ref_pq.push(n);
                REQUIRE(heap.top() == ref_pq.top());
            }
            auto num_pop = seq_dist(gen);
            for (int i = 0; i < num_pop; ++i) {
                if (!heap.empty()) {
                    REQUIRE(heap.top() == ref_pq.top());
                    heap.pop();
                    ref_pq.pop();
                }
            }
        }
        while (!heap.empty()) {
            REQUIRE(!ref_pq.empty());
            REQUIRE(heap.top() == ref_pq.top());
            heap.pop();
            ref_pq.pop();
        }
        REQUIRE(ref_pq.empty());
    }

    SECTION("dijkstra") {
        auto dist = std::uniform_int_distribution{-100, 100};
        auto seq_dist = std::uniform_int_distribution{1, 10};

        heap.push(0);
        ref_pq.push(0);
        for (int s = 0; s < 1000; ++s) {
            auto top = heap.top();
            heap.pop();
            ref_pq.pop();
            auto num_push = seq_dist(gen);
            for (int i = 0; i < num_push; ++i) {
                auto n = top + dist(gen);
                heap.push(n);
                ref_pq.push(n);
                REQUIRE(heap.top() == ref_pq.top());
            }
        }
        while (!heap.empty()) {
            REQUIRE(!ref_pq.empty());
            REQUIRE(heap.top() == ref_pq.top());
            heap.pop();
            ref_pq.pop();
        }
        REQUIRE(ref_pq.empty());
    }
}

TEMPLATE_TEST_CASE("heap handles many equal elements", "[heap][duplicates]",
                   (multiqueue::Heap<int, std::ranges::greater>),
                   (QuickHeap<int, std::ranges::less>),
                   (QuickHeapAVX2<std::int32_t>),
                   (QuickHeapAVX2<std::int64_t>),
                   (QuickHeapAVX2<std::int32_t, true>),
                   (QuickHeapAVX2<std::int64_t, true>),
                   (QuickHeapAVX2<std::int32_t, false, false>),
                   (QuickHeapAVX2<std::int64_t, false, false>),
                   (QuickHeapAVX2<std::int32_t, false, true, Scan::simd>),
                   (QuickHeapAVX2<std::int64_t, false, true, Scan::simd>),
                   (QuickHeapRs<std::int32_t>),
                   (QuickHeapRs<std::int64_t>),
                   (multiqueue::value_merge_heap<int, std::ranges::less>),
                   (MergeHeapAVX2<std::int32_t>),
                   (MergeHeapAVX2<std::int64_t>) AVX512_HEAPS RUST_HEAPS) {
    using heap_t = TestType;

    auto heap = heap_t{};
    auto ref_pq = std::priority_queue<int, std::vector<int>, std::greater<>>{};
    auto gen = std::mt19937{0};

    auto check_drain = [&] {
        while (!heap.empty()) {
            REQUIRE(!ref_pq.empty());
            REQUIRE(heap.top() == ref_pq.top());
            heap.pop();
            ref_pq.pop();
        }
        REQUIRE(ref_pq.empty());
    };

    auto const lowest = std::numeric_limits<int>::lowest();
    auto const [lo, hi] = GENERATE_COPY(table<int, int>(
        {{0, 0}, {0, 1}, {0, 15}, {-1000, 1000}, {lowest, lowest + 3}}));

    SECTION("push all, then pop all") {
        auto dist = std::uniform_int_distribution{lo, hi};
        for (int i = 0; i < 2000; ++i) {
            auto n = dist(gen);
            heap.push(n);
            ref_pq.push(n);
            REQUIRE(heap.top() == ref_pq.top());
        }
        check_drain();
    }

    SECTION("interleave pushing and popping") {
        auto dist = std::uniform_int_distribution{lo, hi};
        auto seq_dist = std::uniform_int_distribution{0, 40};
        for (int s = 0; s < 500; ++s) {
            auto num_push = seq_dist(gen);
            for (int i = 0; i < num_push; ++i) {
                auto n = dist(gen);
                heap.push(n);
                ref_pq.push(n);
                REQUIRE(heap.top() == ref_pq.top());
            }
            auto num_pop = seq_dist(gen);
            for (int i = 0; i < num_pop && !heap.empty(); ++i) {
                REQUIRE(heap.top() == ref_pq.top());
                heap.pop();
                ref_pq.pop();
            }
        }
        check_drain();
    }

    SECTION("monotone pushes at the current minimum") {
        auto step_dist = std::uniform_int_distribution{0, 2};
        heap.push(lo);
        ref_pq.push(lo);
        for (int s = 0; s < 3000; ++s) {
            auto top = heap.top();
            REQUIRE(top == ref_pq.top());
            heap.pop();
            ref_pq.pop();
            for (int i = 0; i < 3; ++i) {
                auto n = top + step_dist(gen);
                heap.push(n);
                ref_pq.push(n);
            }
        }
        check_drain();
    }
}
