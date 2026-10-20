#pragma once

#ifndef __AVX2__
#error "quick_heap_simd.hpp requires AVX2 (e.g. -march=native)"
#endif

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace quick_heap_simd_detail {

inline constexpr std::array<std::uint64_t, 256> compress_lut32 = [] {
    std::array<std::uint64_t, 256> lut = {};
    for (std::uint64_t mask = 0; mask < 256; ++mask) {
        std::uint64_t idx = 0;
        for (std::uint64_t i = 0; i < 8; ++i) {
            if ((mask & (1u << i)) != 0) {
                lut[mask] |= (i << (idx * 8));
                ++idx;
            }
        }
    }
    return lut;
}();

inline constexpr std::array<std::uint64_t, 16> compress_lut64 = [] {
    std::array<std::uint64_t, 16> lut = {};
    for (std::uint64_t mask = 0; mask < 16; ++mask) {
        std::uint64_t idx = 0;
        for (std::uint64_t i = 0; i < 4; ++i) {
            if ((mask & (1u << i)) != 0) {
                lut[mask] |= ((2 * i) << (idx * 8));
                lut[mask] |= ((2 * i + 1) << ((idx + 1) * 8));
                idx += 2;
            }
        }
    }
    return lut;
}();

inline __m256i permute_bytes(__m256i v, std::uint64_t indices) {
    return _mm256_permutevar8x32_epi32(
        v, _mm256_cvtepu8_epi32(
               _mm_cvtsi64_si128(static_cast<std::int64_t>(indices))));
}

template <typename T, bool use_pext = false>
struct Avx2;

template <bool use_pext>
struct Avx2<std::int32_t, use_pext> {
    static constexpr std::size_t L = 8;
    using vec = __m256i;
    using cmp = __m256i;
    static vec splat(std::int32_t x) { return _mm256_set1_epi32(x); }
    static vec load(std::int32_t const *p) {
        return _mm256_loadu_si256(reinterpret_cast<vec const *>(p));
    }
    static void store(std::int32_t *p, vec v) {
        _mm256_storeu_si256(reinterpret_cast<vec *>(p), v);
    }
    static cmp cmpgt(vec a, vec b) { return _mm256_cmpgt_epi32(a, b); }
    static bool all(cmp c) {
        return _mm256_testc_si256(c, _mm256_set1_epi32(-1)) != 0;
    }
    static unsigned mask(cmp c) {
        return static_cast<unsigned>(
            _mm256_movemask_ps(_mm256_castsi256_ps(c)));
    }
    static unsigned gt(vec a, vec b) { return mask(cmpgt(a, b)); }
    static unsigned eq(vec a, vec b) { return mask(_mm256_cmpeq_epi32(a, b)); }
    static vec min(vec a, vec b) { return _mm256_min_epi32(a, b); }
    static std::int32_t hmin(vec v) {
        auto x = _mm_min_epi32(_mm256_castsi256_si128(v),
                               _mm256_extracti128_si256(v, 1));
        x = _mm_min_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));
        x = _mm_min_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1)));
        return _mm_cvtsi128_si32(x);
    }
    // Lanes >= k are replaced by fill.
    static vec load_first(std::int32_t const *p, std::size_t k,
                          std::int32_t fill) {
        auto const keep =
            _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(k)),
                               _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        return _mm256_blendv_epi8(splat(fill), load(p), keep);
    }
    static vec compress(vec v, unsigned mask) {
        assert(mask < 256);
        if constexpr (use_pext) {
            auto const selected =
                _pdep_u64(mask, 0x0101010101010101ull) * 0xffu;
            return permute_bytes(v, _pext_u64(0x0706050403020100ull, selected));
        } else {
            return permute_bytes(v, compress_lut32[mask]);
        }
    }

    static void compress_store(vec v, unsigned mask, std::int32_t *p) {
        store(p, compress(v, mask));
    }
};

template <bool use_pext>
struct Avx2<std::int64_t, use_pext> {
    static constexpr std::size_t L = 4;
    using vec = __m256i;
    using cmp = __m256i;
    static vec splat(std::int64_t x) { return _mm256_set1_epi64x(x); }
    static vec load(std::int64_t const *p) {
        return _mm256_loadu_si256(reinterpret_cast<vec const *>(p));
    }
    static void store(std::int64_t *p, vec v) {
        _mm256_storeu_si256(reinterpret_cast<vec *>(p), v);
    }
    static cmp cmpgt(vec a, vec b) { return _mm256_cmpgt_epi64(a, b); }
    static bool all(cmp c) {
        return _mm256_testc_si256(c, _mm256_set1_epi64x(-1)) != 0;
    }
    static unsigned mask(cmp c) {
        return static_cast<unsigned>(
            _mm256_movemask_pd(_mm256_castsi256_pd(c)));
    }
    static unsigned gt(vec a, vec b) { return mask(cmpgt(a, b)); }
    static unsigned eq(vec a, vec b) { return mask(_mm256_cmpeq_epi64(a, b)); }
    static vec min(vec a, vec b) {
        return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi64(a, b));
    }
    static std::int64_t hmin(vec v) {
        auto const lo = _mm256_castsi256_si128(v);
        auto const hi = _mm256_extracti128_si256(v, 1);
        auto x = _mm_blendv_epi8(lo, hi, _mm_cmpgt_epi64(lo, hi));
        auto const y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        x = _mm_blendv_epi8(x, y, _mm_cmpgt_epi64(x, y));
        return _mm_cvtsi128_si64(x);
    }
    // Lanes >= k are replaced by fill.
    static vec load_first(std::int64_t const *p, std::size_t k,
                          std::int64_t fill) {
        auto const keep =
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(static_cast<long long>(k)),
                               _mm256_setr_epi64x(0, 1, 2, 3));
        return _mm256_blendv_epi8(splat(fill), load(p), keep);
    }
    static vec compress(vec v, unsigned mask) {
        assert(mask < 16);
        if constexpr (use_pext) {
            auto const selected =
                _pdep_u64(mask, 0x0001000100010001ull) * 0xffffu;
            return permute_bytes(v, _pext_u64(0x0706050403020100ull, selected));
        } else {
            return permute_bytes(v, compress_lut64[mask]);
        }
    }

    static void compress_store(vec v, unsigned mask, std::int64_t *p) {
        store(p, compress(v, mask));
    }
};

#ifdef __AVX512F__

template <typename T, bool use_compress_store = false>
struct Avx512;

template <bool use_compress_store>
struct Avx512<std::int32_t, use_compress_store> {
    static constexpr std::size_t L = 16;
    using vec = __m512i;
    using cmp = __mmask16;
    static vec splat(std::int32_t x) { return _mm512_set1_epi32(x); }
    static vec load(std::int32_t const *p) { return _mm512_loadu_si512(p); }
    static void store(std::int32_t *p, vec v) { _mm512_storeu_si512(p, v); }
    static cmp cmpgt(vec a, vec b) { return _mm512_cmpgt_epi32_mask(a, b); }
    static bool all(cmp c) { return c == 0xffffu; }
    static unsigned mask(cmp c) { return c; }
    static unsigned gt(vec a, vec b) { return mask(cmpgt(a, b)); }
    static unsigned eq(vec a, vec b) { return _mm512_cmpeq_epi32_mask(a, b); }
    static vec min(vec a, vec b) { return _mm512_min_epi32(a, b); }
    static std::int32_t hmin(vec v) { return _mm512_reduce_min_epi32(v); }
    static vec load_first(std::int32_t const *p, std::size_t k,
                          std::int32_t fill) {
        return _mm512_mask_loadu_epi32(
            splat(fill), static_cast<__mmask16>((1u << k) - 1u), p);
    }
    static vec compress(vec v, unsigned mask) {
        return _mm512_maskz_compress_epi32(static_cast<__mmask16>(mask), v);
    }
    static void compress_store(vec v, unsigned mask, std::int32_t *p) {
        if constexpr (use_compress_store) {
            _mm512_mask_compressstoreu_epi32(p, static_cast<__mmask16>(mask),
                                             v);
        } else {
            store(p, compress(v, mask));
        }
    }
};

template <bool use_compress_store>
struct Avx512<std::int64_t, use_compress_store> {
    static constexpr std::size_t L = 8;
    using vec = __m512i;
    using cmp = __mmask8;
    static vec splat(std::int64_t x) { return _mm512_set1_epi64(x); }
    static vec load(std::int64_t const *p) { return _mm512_loadu_si512(p); }
    static void store(std::int64_t *p, vec v) { _mm512_storeu_si512(p, v); }
    static cmp cmpgt(vec a, vec b) { return _mm512_cmpgt_epi64_mask(a, b); }
    static bool all(cmp c) { return c == 0xffu; }
    static unsigned mask(cmp c) { return c; }
    static unsigned gt(vec a, vec b) { return mask(cmpgt(a, b)); }
    static unsigned eq(vec a, vec b) { return _mm512_cmpeq_epi64_mask(a, b); }
    static vec min(vec a, vec b) { return _mm512_min_epi64(a, b); }
    static std::int64_t hmin(vec v) { return _mm512_reduce_min_epi64(v); }
    static vec load_first(std::int64_t const *p, std::size_t k,
                          std::int64_t fill) {
        return _mm512_mask_loadu_epi64(
            splat(fill), static_cast<__mmask8>((1u << k) - 1u), p);
    }
    static vec compress(vec v, unsigned mask) {
        return _mm512_maskz_compress_epi64(static_cast<__mmask8>(mask), v);
    }
    static void compress_store(vec v, unsigned mask, std::int64_t *p) {
        if constexpr (use_compress_store) {
            _mm512_mask_compressstoreu_epi64(p, static_cast<__mmask8>(mask), v);
        } else {
            store(p, compress(v, mask));
        }
    }
};

#endif

template <typename S, bool early_exit, typename T>
std::size_t num_greater_pivots(T const *pivots, std::size_t n, T e) {
    auto const e_vec = S::splat(e);
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; i += S::L) {
        auto const c = S::cmpgt(S::load(pivots + i), e_vec);
        if constexpr (early_exit) {
            if (!S::all(c)) {
                return count +
                       static_cast<std::size_t>(std::popcount(S::mask(c)));
            }
            count += S::L;
        } else {
            count += static_cast<std::size_t>(std::popcount(S::mask(c)));
        }
    }
    return count;
}

enum class Scan {
    scalar,
    simd,
};

template <typename T>
T *find_min_scalar(T *data, std::size_t n) {
    return std::min_element(data, data + n);
}

template <typename S, typename T>
T *find_min_simd(T *data, std::size_t n) {
    assert(n > 0);
    auto m = S::splat(std::numeric_limits<T>::max());
    std::size_t i = 0;
    for (; i + S::L <= n; i += S::L) {
        m = S::min(m, S::load(data + i));
    }
    if (i < n) {
        m = S::min(
            m, S::load_first(data + i, n - i, std::numeric_limits<T>::max()));
    }
    auto const min_vec = S::splat(S::hmin(m));
    for (i = 0;; i += S::L) {
        if (auto const mask = S::eq(S::load(data + i), min_vec); mask != 0) {
            return data + i + std::countr_zero(mask);
        }
    }
}

template <typename T, std::size_t Padding>
class Bucket {
    std::unique_ptr<T[]> data_;
    std::size_t size_ = 0;
    std::size_t cap_ = 0;

    void grow(std::size_t n) {
        auto const new_cap = (n + Padding - 1u) & ~(Padding - 1u);
        auto p = std::make_unique_for_overwrite<T[]>(new_cap);
        std::copy_n(data_.get(), size_, p.get());
        data_ = std::move(p);
        cap_ = new_cap;
    }

   public:
    T *data() { return data_.get(); }
    T *begin() { return data(); }
    T *end() { return data() + size_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T const &back() const { return data_[size_ - 1]; }
    void pop_back() { --size_; }
    void clear() { size_ = 0; }

    void push_back(T e) {
        if (size_ == cap_) grow(std::max<std::size_t>(2 * cap_, 2 * Padding));
        data_[size_++] = e;
    }
    void reserve(std::size_t n) {
        if (n > cap_) grow(n);
    }
    void set_size(std::size_t n) {
        assert(n <= cap_);
        size_ = n;
    }
};

template <typename T, typename S, std::size_t N = 16, bool early_exit = true,
          Scan scan = Scan::scalar>
class QuickHeapSimd {
    static_assert(std::is_same_v<T, std::int32_t> ||
                  std::is_same_v<T, std::int64_t>);

    static constexpr std::size_t L = S::L;
    static constexpr unsigned full_mask = (1u << L) - 1u;

   public:
    using value_type = T;

   private:
    std::size_t num_buckets_ = 1;
    std::vector<T> pivots_;
    std::vector<Bucket<T, L>> buckets_;
    // Bucket i holds [pivots_[i], pivots_[i - 1]). The minimum of the last
    // bucket is at its back.

    inline bool last_bucket_is_equal() {
        return num_buckets_ >= 2 &&
               pivots_[num_buckets_ - 2] - 1 == pivots_[num_buckets_ - 1];
    }

    void ensure_pivot_capacity() {
        if (pivots_.size() <= num_buckets_) {
            pivots_.resize(2 * pivots_.size(),
                           std::numeric_limits<T>::lowest());
        }
    }

    static void split(typename S::vec v, unsigned mask_move, unsigned valid,
                      T *&stay_ptr, T *&move_ptr) {
        S::compress_store(v, mask_move, move_ptr);
        move_ptr += std::popcount(mask_move);
        auto const mask_stay = mask_move ^ valid;
        S::compress_store(v, mask_stay, stay_ptr);
        stay_ptr += std::popcount(mask_stay);
    }

    void partition_last_bucket() {
        assert(!last_bucket_is_equal());
        auto const n = buckets_[num_buckets_ - 1].size();
        assert(n >= 1);
        auto const pivot = [&]() {
            auto const a = *buckets_[num_buckets_ - 1].begin();
            auto const b = *(buckets_[num_buckets_ - 1].begin() + n / 2);
            auto const c = *(buckets_[num_buckets_ - 1].end() - 1);
            auto const p =
                std::max(std::min(a, b), std::min(std::max(a, b), c));
            return num_buckets_ >= 2 && p == pivots_[num_buckets_ - 2] - 1
                       ? p
                       : p + 1;
        }();
        if (buckets_.size() <= num_buckets_) {
            buckets_.push_back({});
        }
        buckets_[num_buckets_].reserve(n);
        auto const data = buckets_[num_buckets_ - 1].data();
        auto const pivot_vec = S::splat(pivot);
        auto stay_ptr = buckets_[num_buckets_ - 1].data();
        auto move_ptr = buckets_[num_buckets_].data();
        std::size_t i = 0;
        for (; i + L <= n; i += L) {
            auto const v = S::load(data + i);
            // Also move pivot element (if it's not an equal bucket)
            split(v, S::gt(pivot_vec, v), full_mask, stay_ptr, move_ptr);
        }
        if (i < n) {
            unsigned const valid_mask = (1u << (n - i)) - 1u;
            auto const v = S::load(data + i);
            split(v, S::gt(pivot_vec, v) & valid_mask, valid_mask, stay_ptr,
                  move_ptr);
        }
        buckets_[num_buckets_ - 1].set_size(static_cast<std::size_t>(
            std::distance(buckets_[num_buckets_ - 1].data(), stay_ptr)));
        buckets_[num_buckets_].set_size(static_cast<std::size_t>(
            std::distance(buckets_[num_buckets_].data(), move_ptr)));
        pivots_[num_buckets_ - 1] = pivot;
        if (!buckets_[num_buckets_].empty()) {
            ++num_buckets_;
            ensure_pivot_capacity();
        }
    }

   public:
    explicit QuickHeapSimd() : pivots_(128), buckets_(128) {
        static_assert(128 % L == 0);
        std::fill(pivots_.begin(), pivots_.end(),
                  std::numeric_limits<T>::lowest());
    }

    void push(T e) {
        assert(buckets_.size() >= num_buckets_);
        assert(pivots_.size() % L == 0);
        assert(e != std::numeric_limits<T>::max());
        auto const b =
            num_greater_pivots<S, early_exit>(pivots_.data(), num_buckets_, e);

        assert(b <= num_buckets_);
        if (b == buckets_.size()) {
            buckets_.push_back({});
        }
        auto &bucket = buckets_[b];
        bucket.push_back(e);
        if (b + 1 == num_buckets_) {
            if (bucket.size() >= 2 &&
                *(bucket.end() - 2) < *(bucket.end() - 1)) {
                // No need to check for equal bucket here
                std::iter_swap(bucket.end() - 1, bucket.end() - 2);
            }
            return;
        }
        if (b == num_buckets_) {
            ++num_buckets_;
            ensure_pivot_capacity();
        }
    }

    void pop() {
        assert(num_buckets_ > 0);
        assert(!buckets_[num_buckets_ - 1].empty());
        buckets_[num_buckets_ - 1].pop_back();
        while (num_buckets_ > 1 && buckets_[num_buckets_ - 1].empty()) {
            pivots_[num_buckets_ - 1] = std::numeric_limits<T>::lowest();
            --num_buckets_;
        }
        while (buckets_[num_buckets_ - 1].size() > N &&
               !last_bucket_is_equal()) {
            partition_last_bucket();
        }
        auto &last = buckets_[num_buckets_ - 1];
        if (last.empty() || last_bucket_is_equal()) {
            return;
        }
        auto min_it = scan == Scan::simd
                          ? find_min_simd<S>(last.begin(), last.size())
                          : find_min_scalar(last.begin(), last.size());
        std::iter_swap(min_it, last.end() - 1);
    }

    T const &top() {
        assert(num_buckets_ > 0);
        assert(!buckets_[num_buckets_ - 1].empty());
        return buckets_[num_buckets_ - 1].back();
    }

    bool empty() { return buckets_[num_buckets_ - 1].empty(); }
};

}  // namespace quick_heap_simd_detail

using quick_heap_simd_detail::Scan;

template <typename T, bool use_pext = false, bool early_exit = true,
          Scan scan = Scan::scalar>
using QuickHeapAVX2 = quick_heap_simd_detail::QuickHeapSimd<
    T, quick_heap_simd_detail::Avx2<T, use_pext>, 16, early_exit, scan>;

#ifdef __AVX512F__
template <typename T, bool use_compress_store = false, bool early_exit = true,
          Scan scan = Scan::scalar>
using QuickHeapAVX512 = quick_heap_simd_detail::QuickHeapSimd<
    T, quick_heap_simd_detail::Avx512<T, use_compress_store>, 16, early_exit,
    scan>;
#endif
