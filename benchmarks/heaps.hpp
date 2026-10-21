#pragma once

#include "competitors/merge_heap.hpp"
#include "competitors/merge_heap_simd.hpp"
#include "competitors/mq_heap.hpp"
#include "competitors/quick_heap.hpp"
#include "competitors/quick_heap_rust_port.hpp"
#include "competitors/quick_heap_simd.hpp"
#ifdef WITH_RUST_QUICKHEAP
#include "competitors/quick_heap_rust_wrapper.hpp"
#endif
#include "competitors/s3q/include/s3q/s3q.hpp"

#include <tlx/container/radix_heap.hpp>

#include <boost/heap/d_ary_heap.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/heap/pairing_heap.hpp>

#include <functional>
#include <queue>
#include <utility>
#include <vector>

template <typename T>
using std_pq = std::priority_queue<T, std::vector<T>, std::ranges::greater>;

template <typename T>
using mq_pq = multiqueue::Heap<T, std::ranges::greater>;

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
using merge_heap = multiqueue::value_merge_heap<T, std::ranges::less>;

template <typename T>
using merge_heap_avx2 = MergeHeapAVX2<T>;

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

#ifdef __AVX512F__
template <typename T>
using quick_heap_avx512 = QuickHeapAVX512<T>;
#endif

template <typename T>
using quick_heap_rs = QuickHeapRs<T>;

#ifdef WITH_RUST_QUICKHEAP
template <typename T>
using rust_quick_heap = RustQuickHeap<T>;
#endif

template <typename T>
class s3q_queue {
    struct S3QConfig : s3q::DefaultCfg {
        using Item = T;
    };

   public:
    using value_type = T;

    void push(T item) { pq_.push(std::move(item)); }
    T const& top() const { return pq_.top(); }
    void pop() { pq_.pop(); }

   private:
    s3q::PriorityQueue<S3QConfig> pq_;
};
