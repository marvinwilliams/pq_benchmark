#pragma once

#include "competitors/s3q/include/s3q/s3q.hpp"

#include <utility>

template <typename T>
class s3q_queue {
    struct S3QConfig : s3q::DefaultCfg {
        using Item = T;
    };

    s3q::PriorityQueue<S3QConfig> pq_;

   public:
    using value_type = T;

    void push(T item) { pq_.push(std::move(item)); }
    T const& top() const { return pq_.top(); }
    void pop() { pq_.pop(); }
};
