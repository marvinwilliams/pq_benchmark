#pragma once

// Adapter for the native Rust SimdQuickHeap (../quickheap, `c` feature),
// called through its C ABI via cpp/quickheap.hpp.
//
// The C API only offers push and pop, so top() pops the minimum into a
// one-element cache that the next pop() consumes. A push smaller than the
// cached element swaps places with it, so the cache always holds the minimum.

#include <quickheap.hpp>

#include <cassert>
#include <optional>

template <typename T>
class RustQuickHeap {
    quickheap::Heap<T> heap_;
    std::optional<T> top_;

   public:
    using value_type = T;

    void push(T e) {
        if (top_ && e < *top_) {
            std::swap(e, *top_);
        }
        heap_.push(e);
    }

    T const &top() {
        if (!top_) {
            top_ = heap_.pop();
        }
        return *top_;
    }

    void pop() {
        if (top_) {
            top_.reset();
        } else {
            heap_.pop();
        }
    }

    bool empty() const { return !top_ && heap_.empty(); }
};
