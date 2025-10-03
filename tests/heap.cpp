#include "competitors/merge_heap.hpp"
#include "competitors/mq_heap.hpp"
#include "competitors/quick_heap.hpp"
#include "competitors/quick_heap_avx2.hpp"

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/generators/catch_generators_all.hpp"

#include <array>
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
                   (multiqueue::value_merge_heap<int, std::ranges::less>)) {
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

TEMPLATE_TEST_CASE("max heap",
                   "[heap][comparator]",
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
            REQUIRE(heap.top() ==  999 - i);
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
                   (QuickHeap<int>),
                   (QuickHeapAVX2<std::int32_t>),
                   (multiqueue::value_merge_heap<int, std::ranges::less>)) {
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
            for (int i = 0; i > num_pop; --i) {
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
