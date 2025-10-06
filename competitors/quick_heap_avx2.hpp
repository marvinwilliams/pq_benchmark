#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace quick_heap_avx2_detail {

inline __m256i compress_left(__m256i v, unsigned int mask) {
    auto const mask_expanded = _pdep_u64(mask, 0x0101010101010101) * 0xFF;
    auto const indices =
        static_cast<std::int64_t>(_pext_u64(0x0706050403020100, mask_expanded));
    return _mm256_permutevar8x32_epi32(
        v, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(indices)));
}

inline std::size_t num_greater_pivots(std::int32_t const *pivots, std::size_t n,
                                      std::int32_t e) {
#if defined SKIP_AVX2_POPCNT
    __m256i const ones = _mm256_set1_epi32(-1);
#endif
    __m256i const e_vec = _mm256_set1_epi32(e);
    std::size_t count = 0;
    auto const pivots_end = pivots + n;
    if (n >= 32u) {
        for (; pivots <= pivots_end - 32; pivots += 32) {
            __m256i const data1 =
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(pivots));
            __m256i const data2 = _mm256_loadu_si256(
                reinterpret_cast<__m256i const *>(pivots + 8));
            __m256i const data3 = _mm256_loadu_si256(
                reinterpret_cast<__m256i const *>(pivots + 16));
            __m256i const data4 = _mm256_loadu_si256(
                reinterpret_cast<__m256i const *>(pivots + 24));

            __m256i const cmp1 = _mm256_cmpgt_epi32(data1, e_vec);
            __m256i const cmp2 = _mm256_cmpgt_epi32(data2, e_vec);
            __m256i const cmp3 = _mm256_cmpgt_epi32(data3, e_vec);
            __m256i const cmp4 = _mm256_cmpgt_epi32(data4, e_vec);
#if defined SKIP_AVX2_POPCNT
            __m256i const comb1 = _mm256_and_si256(cmp1, cmp2);
            __m256i const comb2 = _mm256_and_si256(cmp3, cmp4);
            __m256i const comb3 = _mm256_and_si256(comb1, comb2);
            if (_mm256_testc_si256(comb3, ones)) {
                count += 32;
            } else {
#endif
                auto const count1 = static_cast<std::size_t>(
                    _mm_popcnt_u32(static_cast<unsigned>(
                        _mm256_movemask_ps(_mm256_castsi256_ps(cmp1)))));
                auto const count2 = static_cast<std::size_t>(
                    _mm_popcnt_u32(static_cast<unsigned>(
                        _mm256_movemask_ps(_mm256_castsi256_ps(cmp2)))));
                auto const count3 = static_cast<std::size_t>(
                    _mm_popcnt_u32(static_cast<unsigned>(
                        _mm256_movemask_ps(_mm256_castsi256_ps(cmp3)))));
                auto const count4 = static_cast<std::size_t>(
                    _mm_popcnt_u32(static_cast<unsigned>(
                        _mm256_movemask_ps(_mm256_castsi256_ps(cmp4)))));
                count += count1 + count2 + count3 + count4;
#if defined SKIP_AVX2_POPCNT
                return count;
            }
#endif
        }
    }
    for (; pivots <= pivots_end - 8; pivots += 8) {
        __m256i const data =
            _mm256_loadu_si256(reinterpret_cast<__m256i const *>(pivots));
        __m256i const cmp = _mm256_cmpgt_epi32(data, e_vec);
#if defined SKIP_AVX2_POPCNT
        if (_mm256_testc_si256(cmp, ones)) {
            count += 8;
        } else {
#endif
            count += static_cast<std::size_t>(
                _mm_popcnt_u32(static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(cmp)))));
#if defined SKIP_AVX2_POPCNT
            return count;
        }
#endif
    }
    auto const remaining = static_cast<int>(pivots_end - pivots);
    assert(remaining >= 0 && remaining < 8);
    if (remaining > 0) {
        __m256i const data =
            _mm256_loadu_si256(reinterpret_cast<__m256i const *>(pivots));
        __m256i const cmp = _mm256_cmpgt_epi32(data, e_vec);
        count += static_cast<std::size_t>(
            _mm_popcnt_u32(static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(cmp))) &
                           ((1u << remaining) - 1u)));
    }
    return count;
}

inline std::size_t find_bucket(std::int32_t const *pivots, std::size_t n,
                               std::int32_t e) {
    auto const num_greater = num_greater_pivots(pivots, n, e);
    return (num_greater == static_cast<std::size_t>(n))
               ? (n * 2)
               : (num_greater * 2 +
                  static_cast<std::size_t>(pivots[num_greater] == e));
}

template <typename It>
It find_min(It data, int n) {
    return std::min_element(data, data + n);
}

template <typename T>
class QuickHeapAVX2 {
    static_assert(std::is_same_v<T, std::int32_t> ||
                  std::is_same_v<T, std::int64_t>);

   public:
    using value_type = T;

   private:
    std::size_t num_buckets_ = 1;
    std::vector<T> pivots_;
    std::vector<std::vector<T>> buckets_;

    void partition_last_bucket() {
        assert((num_buckets_ & 1) == 1);
        auto const n = buckets_[num_buckets_ - 1].size();
        assert(n >= 1);
        auto const a = buckets_[num_buckets_ - 1][0];
        auto const b = buckets_[num_buckets_ - 1][n / 2];
        auto const c = buckets_[num_buckets_ - 1][n - 1];
        auto const pivot =
            std::max(std::min(a, b), std::min(std::max(a, b), c));
        if (buckets_.size() <= num_buckets_ + 1) {
            buckets_.insert(buckets_.end(), 2, {});
        }
        buckets_[num_buckets_ + 1].resize((n + 7u) & ~7u);
        buckets_[num_buckets_].resize((n + 7u) & ~7u);
        buckets_[num_buckets_ - 1].resize((n + 7u) & ~7u);
        auto data = buckets_[num_buckets_ - 1].data();
        auto const data_end = data + n;
        __m256i const pivot_vec = _mm256_set1_epi32(pivot);
        auto smaller_ptr = buckets_[num_buckets_ + 1].data();
        auto equal_ptr = buckets_[num_buckets_].data();
        auto greater_ptr = buckets_[num_buckets_ - 1].data();
        for (; data <= data_end - 8; data += 8) {
            __m256i const v =
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(data));
            auto const mask_smaller = static_cast<unsigned>(_mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(pivot_vec, v))));
            auto const mask_greater = static_cast<unsigned>(_mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(v, pivot_vec))));
            auto const mask_equal = ~(mask_smaller | mask_greater) & 0xFF;
            auto const compressed_smaller = compress_left(v, mask_smaller);
            auto const compressed_greater = compress_left(v, mask_greater);
            auto const compressed_equal = compress_left(v, mask_equal);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(smaller_ptr),
                                compressed_smaller);
            smaller_ptr += _mm_popcnt_u32(mask_smaller);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(greater_ptr),
                                compressed_greater);
            greater_ptr += _mm_popcnt_u32(mask_greater);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(equal_ptr),
                                compressed_equal);
            equal_ptr += _mm_popcnt_u32(mask_equal);
        }
        auto const remaining = data_end - data;
        assert(remaining >= 0 && remaining < 8);
        if (remaining != 0) {
            unsigned const valid_mask = (1u << remaining) - 1u;
            __m256i const v =
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(data));
            auto const mask_smaller =
                static_cast<unsigned>(_mm256_movemask_ps(
                    _mm256_castsi256_ps(_mm256_cmpgt_epi32(pivot_vec, v)))) &
                valid_mask;
            auto const mask_greater =
                static_cast<unsigned>(_mm256_movemask_ps(
                    _mm256_castsi256_ps(_mm256_cmpgt_epi32(v, pivot_vec)))) &
                valid_mask;
            auto const mask_equal = ~(mask_smaller | mask_greater) & valid_mask;
            auto const compressed_smaller = compress_left(v, mask_smaller);
            auto const compressed_greater = compress_left(v, mask_greater);
            auto const compressed_equal = compress_left(v, mask_equal);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(smaller_ptr),
                                compressed_smaller);
            smaller_ptr += _mm_popcnt_u32(mask_smaller);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(greater_ptr),
                                compressed_greater);
            greater_ptr += _mm_popcnt_u32(mask_greater);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(equal_ptr),
                                compressed_equal);
            equal_ptr += _mm_popcnt_u32(mask_equal);
        }
        buckets_[num_buckets_ + 1].resize(static_cast<std::size_t>(
            std::distance(buckets_[num_buckets_ + 1].data(), smaller_ptr)));
        buckets_[num_buckets_].resize(static_cast<std::size_t>(
            std::distance(buckets_[num_buckets_].data(), equal_ptr)));
        buckets_[num_buckets_ - 1].resize(static_cast<std::size_t>(
            std::distance(buckets_[num_buckets_ - 1].data(), greater_ptr)));
        assert(!buckets_[num_buckets_].empty());
        if (pivots_.size() <= num_buckets_ / 2) {
            pivots_.insert(pivots_.end(), 8, 0);
        }
        pivots_[num_buckets_ / 2] = pivot;
        num_buckets_ += (buckets_[num_buckets_ + 1].empty() ? 1u : 2u);
    }

   public:
    explicit QuickHeapAVX2() : pivots_(128), buckets_(128) {}

    void push(T e) {
        assert(buckets_.size() >= (num_buckets_ | 1));
        assert((pivots_.size() & 7) == 0);
        auto const b = find_bucket(pivots_.data(), num_buckets_ / 2, e);
        assert(b < (num_buckets_ | 1));
        auto &bucket = buckets_[b];
        bucket.push_back(e);
        if (bucket.capacity() < ((bucket.size() + 7u) & ~7u)) {
            auto const old_size = bucket.size();
            bucket.insert(bucket.end(), 8u - (bucket.size() & 7), 0);
            bucket.resize(old_size);
        }
        if (b == num_buckets_ - 1 && ((b & 1) == 0) && bucket.size() > 1 &&
            *(bucket.end() - 2) < e) {
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
               (buckets_[num_buckets_ - 1].size() > 256 / (8 * sizeof(T)))) {
            partition_last_bucket();
        }
        if ((num_buckets_ & 1) == 1 && !buckets_[num_buckets_ - 1].empty()) {
            auto m =
                find_min(buckets_[num_buckets_ - 1].begin(),
                         static_cast<int>(buckets_[num_buckets_ - 1].size()));
            std::iter_swap(m, buckets_[num_buckets_ - 1].end() - 1);
        }
    }

    T const &top() {
        assert(num_buckets_ > 0);
        assert(!buckets_[num_buckets_ - 1].empty());
        return buckets_[num_buckets_ - 1].back();
    }

    bool empty() { return buckets_[num_buckets_ - 1].empty(); }
};

}  // namespace quick_heap_avx2_detail

template <typename T>
using QuickHeapAVX2 = quick_heap_avx2_detail::QuickHeapAVX2<T>;
