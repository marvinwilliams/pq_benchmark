#pragma once

// Port of the Rust SimdQuickHeap (../quickheap, src/lib.rs + src/simd.rs) with
// its default configuration: AVX2 backend, MedianOfM<3> pivots, no
// rebalancing, N = 16, SORT = true.
//
// Layer i holds elements in [pivots[i], pivots[i - 1]]; elements equal to a
// pivot can be on either side. Ties are split by position during partitioning:
// lanes before the pivot's position go down on <=, lanes after it on <.

#ifndef __AVX2__
#error "quick_heap_rust_port.hpp requires AVX2 (e.g. -march=native)"
#endif

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace quick_heap_rs_detail {

// Growable buffer whose capacity may be read and written beyond size(), like
// the Rust code does with Vec::set_len.
template <typename T>
class Buffer {
    std::unique_ptr<T[]> data_;
    std::size_t size_ = 0;
    std::size_t cap_ = 0;

   public:
    T *data() { return data_.get(); }
    T *begin() { return data(); }
    T *end() { return data() + size_; }
    std::size_t size() const { return size_; }
    std::size_t capacity() const { return cap_; }
    bool empty() const { return size_ == 0; }
    T back() const { return data_[size_ - 1]; }
    void clear() { size_ = 0; }
    void set_size(std::size_t n) {
        assert(n <= cap_);
        size_ = n;
    }

    // Like Vec::reserve: capacity >= size() + additional, amortized growth.
    void reserve(std::size_t additional) {
        auto const required = size_ + additional;
        if (required <= cap_) {
            return;
        }
        auto const new_cap =
            std::max({2 * cap_, required, std::size_t{4}});
        auto p = std::make_unique_for_overwrite<T[]>(new_cap);
        if (size_ != 0) {
            std::memcpy(p.get(), data_.get(), size_ * sizeof(T));
        }
        data_ = std::move(p);
        cap_ = new_cap;
    }

    void push_back(T e) {
        reserve(1);
        data_[size_++] = e;
    }
    T pop_back() { return data_[--size_]; }
    void insert(std::size_t pos, T e) {
        reserve(1);
        std::memmove(data_.get() + pos + 1, data_.get() + pos,
                     (size_ - pos) * sizeof(T));
        data_[pos] = e;
        ++size_;
    }
    T swap_remove(std::size_t pos) {
        auto const e = data_[pos];
        data_[pos] = data_[--size_];
        return e;
    }
};

// UNIQSHUF32[m] compresses the lanes *not* set in m to the front.
inline constexpr auto uniqshuf32 = [] {
    std::array<std::array<std::int32_t, 8>, 256> table{};
    for (unsigned m = 0; m < 256; ++m) {
        unsigned k = 0;
        for (unsigned i = 0; i < 8; ++i) {
            if ((m & (1u << i)) == 0) {
                table[m][k++] = static_cast<std::int32_t>(i);
            }
        }
    }
    return table;
}();

// UNIQSHUF64[m] compresses the 64-bit lanes not set in m (as 32-bit pairs).
inline constexpr auto uniqshuf64 = [] {
    std::array<std::array<std::int32_t, 8>, 16> table{};
    for (unsigned m = 0; m < 16; ++m) {
        unsigned k = 0;
        for (unsigned i = 0; i < 4; ++i) {
            if ((m & (1u << i)) == 0) {
                table[m][k++] = static_cast<std::int32_t>(2 * i);
                table[m][k++] = static_cast<std::int32_t>(2 * i + 1);
            }
        }
    }
    return table;
}();

template <typename T>
struct Avx2;

template <>
struct Avx2<std::int32_t> {
    static constexpr std::size_t L = 8;
    static constexpr unsigned full = 0xFF;
    static __m256i splat(std::int32_t v) { return _mm256_set1_epi32(v); }
    // Bit i set iff a[i] > b[i].
    static unsigned gt(__m256i a, __m256i b) {
        return static_cast<unsigned>(
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a, b))));
    }
    static __m256i compress_excluding(__m256i v, unsigned exclude) {
        auto const key = _mm256_loadu_si256(
            reinterpret_cast<__m256i const *>(uniqshuf32[exclude].data()));
        return _mm256_permutevar8x32_epi32(v, key);
    }
};

template <>
struct Avx2<std::int64_t> {
    static constexpr std::size_t L = 4;
    static constexpr unsigned full = 0xF;
    static __m256i splat(std::int64_t v) { return _mm256_set1_epi64x(v); }
    static unsigned gt(__m256i a, __m256i b) {
        return static_cast<unsigned>(
            _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(a, b))));
    }
    static __m256i compress_excluding(__m256i v, unsigned exclude) {
        auto const key = _mm256_loadu_si256(
            reinterpret_cast<__m256i const *>(uniqshuf64[exclude].data()));
        return _mm256_permutevar8x32_epi32(v, key);
    }
};

template <typename T>
__m256i load(T const *p) {
    return _mm256_loadu_si256(reinterpret_cast<__m256i const *>(p));
}

template <typename T>
void store(T *p, __m256i v) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v);
}

// Writes the lanes in `large` to v and those in `small` to w.
template <typename T>
void write_split(__m256i vals, unsigned small, unsigned large, T *v,
                 std::size_t &v_idx, T *w, std::size_t &w_idx) {
    using S = Avx2<T>;
    store(v + v_idx, S::compress_excluding(vals, ~large & S::full));
    v_idx += static_cast<std::size_t>(std::popcount(large));
    store(w + w_idx, S::compress_excluding(vals, ~small & S::full));
    w_idx += static_cast<std::size_t>(std::popcount(small));
}

// EQUAL_DOWN = false: lanes < threshold go to w, >= threshold to v.
// EQUAL_DOWN = true:  lanes <= threshold go to w, > threshold to v.
template <bool EqualDown, typename T>
void partition_fast(__m256i vals, __m256i threshold, T *v, std::size_t &v_idx,
                    T *w, std::size_t &w_idx) {
    using S = Avx2<T>;
    unsigned const small = EqualDown ? (~S::gt(vals, threshold) & S::full)
                                     : S::gt(threshold, vals);
    write_split(vals, small, small ^ S::full, v, v_idx, w, w_idx);
}

// Like partition_fast<false>, but only the first `len` lanes are valid.
template <typename T>
void partition_slow(__m256i vals, std::size_t len, __m256i threshold, T *v,
                    std::size_t &v_idx, T *w, std::size_t &w_idx) {
    using S = Avx2<T>;
    unsigned const in_range = (1u << len) - 1u;
    unsigned const small = S::gt(threshold, vals) & in_range;
    unsigned const large = ~small & in_range;
    write_split(vals, small, large, v, v_idx, w, w_idx);
}

// Number of pivots p with t < p (pivots are decreasing). Reads up to L - 1
// elements past the end, which must be within capacity.
template <typename T>
std::size_t push_position(T const *pivots, std::size_t n, T t) {
    using S = Avx2<T>;
    if (n <= 64) {
        auto const t_simd = S::splat(t);
        std::size_t target = 0;
        for (std::size_t i = 0; i < n; i += S::L) {
            auto const lt = S::gt(load(pivots + i), t_simd);
            target += static_cast<std::size_t>(std::countr_one(lt));
        }
        return std::min(target, n);
    }
    return static_cast<std::size_t>(
        std::partition_point(pivots, pivots + n,
                             [t](T p) { return !(p < t); }) -
        pivots);
}

// Scalar min search with two interleaved accumulators.
template <typename T>
std::size_t position_min(T const *v, std::size_t n) {
    std::size_t min_pos[2] = {0, 0};
    T min_val[2] = {std::numeric_limits<T>::max(),
                    std::numeric_limits<T>::max()};
    for (std::size_t i = 0; i + 2 <= n; i += 2) {
        if (v[i] < min_val[0]) {
            min_val[0] = v[i];
            min_pos[0] = i;
        }
        if (v[i + 1] < min_val[1]) {
            min_val[1] = v[i + 1];
            min_pos[1] = i + 1;
        }
    }
    if (n % 2 == 1 && v[n - 1] < min_val[0]) {
        min_val[0] = v[n - 1];
        min_pos[0] = n - 1;
    }
    return min_val[0] <= min_val[1] ? min_pos[0] : min_pos[1];
}

// Stand-in for rand::random_range (the Rust version uses the thread RNG).
class Rng {
    std::uint64_t state_ = 0x9E3779B97F4A7C15ull;

   public:
    std::size_t random_range(std::size_t n) {
        state_ += 0xA0761D6478BD642Full;
        auto const x = static_cast<unsigned __int128>(state_) *
                       (state_ ^ 0xE7037ED1A0B428DBull);
        auto const r = static_cast<std::uint64_t>(x >> 64) ^
                       static_cast<std::uint64_t>(x);
        return static_cast<std::size_t>(
            (static_cast<unsigned __int128>(r) * n) >> 64);
    }
};

template <typename T, std::size_t N = 16, bool Sort = true>
class QuickHeapRs {
    static_assert(std::is_same_v<T, std::int32_t> ||
                  std::is_same_v<T, std::int64_t>);
    using S = Avx2<T>;
    static constexpr std::size_t L = S::L;

   public:
    using value_type = T;

   private:
    // Decreasing pivots; capacity always exceeds size() + L for SIMD reads.
    Buffer<T> pivots_;
    std::vector<Buffer<T>> buckets_;
    std::size_t size_ = 0;
    Rng rng_;

    // MedianOfM<3>: median by (value, position) of three random samples.
    std::pair<T, std::size_t> pick_pivot(Buffer<T> &layer) {
        auto const n = layer.size();
        std::array<std::pair<T, std::size_t>, 3> s;
        for (auto &e : s) {
            auto const pos = rng_.random_range(n);
            e = {layer.data()[pos], pos};
        }
        std::sort(s.begin(), s.end());
        return s[1];
    }

    static void sort_decreasing(Buffer<T> &layer) {
        std::sort(layer.begin(), layer.end(), std::greater<>{});
    }

    static T wrapping_add_one(T t) {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<U>(t) + 1u));
    }

    __attribute__((noinline)) void partition() {
        auto const layer = pivots_.size();
        if (layer + 2 * L >= pivots_.capacity()) {
            pivots_.reserve(L);
        }
        if (layer + 1 == buckets_.size()) {
            buckets_.emplace_back();
        }
        auto &cur = buckets_[layer];
        auto &next = buckets_[layer + 1];
        auto const n = cur.size();

        auto const [pivot, pivot_pos] = pick_pivot(cur);
        pivots_.push_back(pivot);

        cur.reserve(L);
        next.clear();
        next.reserve(n + L);

        auto const n2 = std::max((n + L - 1) / L * L, L) - L;

        std::size_t cur_len = 0;
        std::size_t next_len = 0;
        auto const half = (std::min(pivot_pos + 1, n2) + L - 1) / L * L;
        auto const threshold = S::splat(pivot);
        T *const c = cur.data();
        T *const w = next.data();
        for (std::size_t i = 0; i < half; i += L) {
            partition_fast<true>(load(c + i), threshold, c, cur_len, w,
                                 next_len);
        }
        for (std::size_t i = half; i < n2; i += L) {
            partition_fast<false>(load(c + i), threshold, c, cur_len, w,
                                  next_len);
        }
        if (n2 < n) {
            auto const t = pivot_pos >= n2 ? S::splat(wrapping_add_one(pivot))
                                           : threshold;
            partition_slow(load(c + n2), n - n2, t, c, cur_len, w, next_len);
        }

        assert(next_len > 0);
        cur.set_size(cur_len);
        next.set_size(next_len);

        // If all elements went to the next layer because the pivot was the
        // largest one, undo and try again.
        if (cur_len == 0) {
            std::swap(cur, next);
            pivots_.pop_back();
        }
    }

    // Splits the last layer until it is small, then sorts it (decreasing).
    // Idempotent, so both top() and pop() call it.
    void prepare() {
        if (buckets_[pivots_.size()].size() > N) {
            while (buckets_[pivots_.size()].size() > N) {
                partition();
            }
            if constexpr (Sort) {
                sort_decreasing(buckets_[pivots_.size()]);
            }
        }
    }

   public:
    QuickHeapRs() : buckets_(128) { pivots_.reserve(128); }

    void push(T t) {
        auto const target = push_position(pivots_.data(), pivots_.size(), t);
        auto &layer = buckets_[target];
        layer.reserve(L + 1);
        if (Sort && target == pivots_.size() && layer.size() < N) {
            auto const pos = std::partition_point(layer.begin(), layer.end(),
                                                  [t](T x) { return x > t; });
            layer.insert(static_cast<std::size_t>(pos - layer.begin()), t);
        } else {
            layer.push_back(t);
        }
        ++size_;
    }

    T const &top() {
        assert(!empty());
        prepare();
        auto &layer = buckets_[pivots_.size()];
        if constexpr (Sort) {
            return layer.data()[layer.size() - 1];
        } else {
            return layer.data()[position_min(layer.data(), layer.size())];
        }
    }

    void pop() {
        assert(!empty());
        prepare();
        auto &layer = buckets_[pivots_.size()];
        if constexpr (Sort) {
            layer.pop_back();
        } else {
            layer.swap_remove(position_min(layer.data(), layer.size()));
        }
        if (layer.empty() && pivots_.size() > 0) {
            pivots_.pop_back();
            if (Sort && buckets_[pivots_.size()].size() <= N) {
                sort_decreasing(buckets_[pivots_.size()]);
            }
        }
        --size_;
    }

    bool empty() const { return size_ == 0; }
    std::size_t size() const { return size_; }
};

}  // namespace quick_heap_rs_detail

template <typename T, std::size_t N = 16, bool Sort = true>
using QuickHeapRs = quick_heap_rs_detail::QuickHeapRs<T, N, Sort>;
