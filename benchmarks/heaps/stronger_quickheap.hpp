#pragma once

#include "competitors/quickheap/MinRandQH2.hpp"

#include <optional>
#include <utility>

template <typename T>
class stronger_quickheap {
    MinRandQH2<T> pq_{1 << 28};
    std::optional<T> top_;

   public:
    using value_type = T;

    void push(T e) {
        if (top_ && e < *top_) {
            std::swap(e, *top_);
        }
        pq_.insert(e);
    }

    T const& top() {
        if (!top_) {
            top_ = pq_.extractMin();
        }
        return *top_;
    }

    void pop() {
        if (top_) {
            top_.reset();
        } else {
            pq_.extractMin();
        }
    }

    bool empty() const { return !top_ && pq_.isEmpty(); }
};
