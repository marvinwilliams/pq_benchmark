#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace quick_heap_detail {

template <typename It, typename T, typename Comparator>
std::size_t find_bucket(It pivots, int n, T e, Comparator comp) {
    assert(n >= 0);
    int i = 0;
    while (i < n && comp(e, *pivots)) {
        ++i;
        ++pivots;
    }
    if (i == n) {
        return 2 * n;
    }
    return 2 * i - static_cast<std::size_t>(comp(e, *pivots));
}

template <typename It, typename Comparator>
It find_min(It data, int n, Comparator comp) {
    return std::min_element(data, data + n, comp);
}

}  // namespace quick_heap_detail

template <typename T, typename Comparator = std::ranges::less,
          std::size_t PartitionThreshold = 16>
class QuickHeap {
   public:
    using value_type = T;
    using comparator_type = Comparator;

   private:
    std::size_t num_buckets_ = 1;
    std::vector<T> pivots_;
    std::vector<std::vector<T>> buckets_;
    Comparator comp_;

    void partition_last_bucket() {
        assert((num_buckets_ & 1) == 1);
        assert(buckets_[num_buckets_ - 1].size() >= 1);
        auto const n = buckets_[num_buckets_ - 1].size();
        auto const a = buckets_[num_buckets_ - 1][0];
        auto const b = buckets_[num_buckets_ - 1][n / 2];
        auto const c = buckets_[num_buckets_ - 1][n - 1];
        auto const pivot =
            std::max(std::min(a, b), std::min(std::max(a, b), c));
        if (buckets_.size() <= num_buckets_ + 1) {
            buckets_.push_back({});
            buckets_.push_back({});
            pivots_.push_back(0);
        }
        buckets_[num_buckets_ + 1].resize(n);
        buckets_[num_buckets_].resize(n);
        auto smaller_it = buckets_[num_buckets_ + 1].begin();
        auto equal_it = buckets_[num_buckets_].begin();
        auto larger_it = buckets_[num_buckets_ - 1].begin();
        for (auto e : buckets_[num_buckets_ - 1]) {
            if (comp_(e, pivot)) {
                *(smaller_it++) = std::move(e);
            } else if (comp_(pivot, e)) {
                *(larger_it++) = std::move(e);
            } else {
                *(equal_it++) = std::move(e);
            }
        }
        buckets_[num_buckets_ + 1].resize(
            std::distance(buckets_[num_buckets_ + 1].begin(), smaller_it));
        buckets_[num_buckets_].resize(
            std::distance(buckets_[num_buckets_].begin(), equal_it));
        buckets_[num_buckets_ - 1].resize(
            std::distance(buckets_[num_buckets_ - 1].begin(), larger_it));
        assert(!buckets_[num_buckets_].empty());
        pivots_[num_buckets_ / 2] = pivot;
        num_buckets_ += (buckets_[num_buckets_ + 1].empty() ? 1 : 2);
    }

   public:
    explicit QuickHeap(Comparator comp = {})
        : buckets_(128), pivots_(128), comp_(comp) {}

    void push(T e) {
        assert(buckets_.size() >= (num_buckets_ | 1));
        auto const b = quick_heap_detail::find_bucket(
            pivots_.data(), num_buckets_ / 2, e, comp_);
        assert(b < (num_buckets_ | 1));
        auto& bucket = buckets_[b];
        bucket.push_back(e);
        if (b == num_buckets_ - 1 && ((b & 1) == 0) && bucket.size() > 1 &&
            comp_(*(bucket.end() - 2), e)) {
            std::iter_swap(bucket.end() - 1, bucket.end() - 2);
        }
        if (b == num_buckets_) {
            assert((num_buckets_ & 1) == 0);
            ++num_buckets_;
        }
    }

    void pop() {
        assert(num_buckets_ > 0);
        assert(!buckets_[num_buckets_ - 1].empty());
        buckets_[num_buckets_ - 1].pop_back();
        if (buckets_[num_buckets_ - 1].empty() && num_buckets_ > 1) {
            --num_buckets_;
            if (buckets_[num_buckets_ - 1].empty() && num_buckets_ > 1) {
                --num_buckets_;
            }
        }
        while (((num_buckets_ & 1) == 1) &&
               (buckets_[num_buckets_ - 1].size() > PartitionThreshold)) {
            partition_last_bucket();
        }
        if ((num_buckets_ & 1) == 1 && !buckets_[num_buckets_ - 1].empty()) {
            auto m = quick_heap_detail::find_min(
                buckets_[num_buckets_ - 1].begin(),
                static_cast<int>(buckets_[num_buckets_ - 1].size()), comp_);
            std::iter_swap(m, buckets_[num_buckets_ - 1].end() - 1);
        }
    }

    T const& top() {
        assert(num_buckets_ > 0);
        assert(!buckets_[num_buckets_ - 1].empty());
        return buckets_[num_buckets_ - 1].back();
    }

    bool empty() { return buckets_[num_buckets_ - 1].empty(); }
};
