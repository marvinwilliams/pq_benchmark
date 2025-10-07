#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#define SKIP_AVX2_POPCNT
#define WITH_LUT

namespace quick_heap_avx2_no_equal_detail {

#ifdef WITH_LUT

static constexpr std::array<std::uint64_t, 256> compress_lut = [] {
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

inline __m256i compress_left(__m256i v, unsigned int mask) {
    assert(mask < 256);
    auto const indices = static_cast<std::int64_t>(compress_lut[mask]);
    return _mm256_permutevar8x32_epi32(
        v, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(indices)));
}

#else

inline __m256i compress_left(__m256i v, unsigned int mask) {
    auto const mask_expanded = _pdep_u64(mask, 0x0101010101010101) * 0xFF;
    auto const indices =
        static_cast<std::int64_t>(_pext_u64(0x0706050403020100, mask_expanded));
    return _mm256_permutevar8x32_epi32(
        v, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(indices)));
}

#endif

inline std::size_t num_greater_pivots(std::int32_t const *pivots, std::size_t n,
                                      std::int32_t e) {
#ifdef SKIP_AVX2_POPCNT
    __m256i const ones = _mm256_set1_epi32(-1);
#endif
    __m256i const e_vec = _mm256_set1_epi32(e);
    std::size_t count = 0;
    auto const pivots_end = pivots + n;
    if (pivots_end - pivots >= 32) {
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
#ifdef SKIP_AVX2_POPCNT
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
#ifdef SKIP_AVX2_POPCNT
                return count;
            }
#endif
        }
    }
    if (pivots_end - pivots >= 8) {
        for (; pivots <= pivots_end - 8; pivots += 8) {
            __m256i const data =
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(pivots));
            __m256i const cmp = _mm256_cmpgt_epi32(data, e_vec);
#ifdef SKIP_AVX2_POPCNT
            if (_mm256_testc_si256(cmp, ones)) {
                count += 8;
            } else {
#endif
                count += static_cast<std::size_t>(
                    _mm_popcnt_u32(static_cast<unsigned>(
                        _mm256_movemask_ps(_mm256_castsi256_ps(cmp)))));
#ifdef SKIP_AVX2_POPCNT
                return count;
            }
#endif
        }
    }
    auto const remaining = static_cast<int>(pivots_end - pivots);
    assert(remaining >= 0 && remaining < 8);
    if (remaining > 0) {
        __m256i const data =
            _mm256_loadu_si256(reinterpret_cast<__m256i const *>(pivots));
        __m256i const cmp = _mm256_cmpgt_epi32(data, e_vec);
        count += static_cast<std::size_t>(
            _mm_popcnt_u32(static_cast<unsigned>(
                               _mm256_movemask_ps(_mm256_castsi256_ps(cmp))) &
                           ((1u << remaining) - 1u)));
    }
    return count;
}

inline std::size_t find_bucket(std::int32_t const *pivots, std::size_t n,
                               std::int32_t e) {
    return num_greater_pivots(pivots, n, e);
}

template <typename It>
It find_min(It data, int n) {
    return std::min_element(data, data + n);
}

template <typename T>
class QuickHeapAVX2NoEqual {
    static_assert(std::is_same_v<T, std::int32_t> ||
                  std::is_same_v<T, std::int64_t>);

   public:
    using value_type = T;

   private:
    std::size_t num_buckets_ = 1;
    std::vector<T> pivots_;
    std::vector<std::vector<T>> buckets_;

    void partition_last_bucket() {
        auto const n = buckets_[num_buckets_ - 1].size();
        assert(n >= 1);
        auto const a = buckets_[num_buckets_ - 1][0];
        auto const b = buckets_[num_buckets_ - 1][n / 2];
        auto const c = buckets_[num_buckets_ - 1][n - 1];
        auto const pivot =
            std::max(std::min(a, b), std::min(std::max(a, b), c));
        if (buckets_.size() <= num_buckets_) {
            buckets_.push_back({});
        }
        buckets_[num_buckets_].resize((n + 7u) & ~7u);
        buckets_[num_buckets_ - 1].resize((n + 7u) & ~7u);
        auto data = buckets_[num_buckets_ - 1].data();
        auto const data_end = data + n;
        __m256i const pivot_vec = _mm256_set1_epi32(pivot);
        auto smaller_ptr = buckets_[num_buckets_].data();
        auto greater_ptr = buckets_[num_buckets_ - 1].data();
        for (; data <= data_end - 8; data += 8) {
            __m256i const v =
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(data));
            auto const mask_smaller = static_cast<unsigned>(_mm256_movemask_ps(
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(pivot_vec, v))));
            auto const num_smaller = _mm_popcnt_u32(mask_smaller);
            auto const mask_greater = ~mask_smaller & 0xFF;
            auto const compressed_smaller = compress_left(v, mask_smaller);
            auto const compressed_greater = compress_left(v, mask_greater);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(smaller_ptr),
                                compressed_smaller);
            smaller_ptr += num_smaller;
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(greater_ptr),
                                compressed_greater);
            greater_ptr += 8 - num_smaller;
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
            auto const num_smaller = _mm_popcnt_u32(mask_smaller);
            auto const mask_greater = (~mask_smaller) & valid_mask;
            auto const compressed_smaller = compress_left(v, mask_smaller);
            auto const compressed_greater = compress_left(v, mask_greater);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(smaller_ptr),
                                compressed_smaller);
            smaller_ptr += num_smaller;
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(greater_ptr),
                                compressed_greater);
            greater_ptr += remaining - num_smaller;
        }
        assert(std::distance(buckets_[num_buckets_ - 1].data(), greater_ptr) >
               0);  // must have the pivot
        if (smaller_ptr == buckets_[num_buckets_].data()) {
            assert(greater_ptr == buckets_[num_buckets_ - 1].data() + n);
            data = buckets_[num_buckets_ - 1].data();
            if (std::all_of(data, data + n,
                            [pivot](T x) { return x == pivot; })) {
                smaller_ptr = std::fill_n(smaller_ptr, n / 2, pivot);
                greater_ptr -= n / 2;
            } else {
                greater_ptr = buckets_[num_buckets_ - 1].data();
                for (; data <= data_end - 8; data += 8) {
                    __m256i const v = _mm256_loadu_si256(
                        reinterpret_cast<__m256i const *>(data));
                    auto const mask_equal = static_cast<unsigned>(
                        _mm256_movemask_ps(_mm256_castsi256_ps(
                            _mm256_cmpeq_epi32(pivot_vec, v))));
                    auto const num_equal = _mm_popcnt_u32(mask_equal);
                    auto const mask_greater = ~mask_equal & 0xFF;
                    auto const compressed_equal = compress_left(v, mask_equal);
                    auto const compressed_greater =
                        compress_left(v, mask_greater);
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(smaller_ptr),
                        compressed_equal);
                    smaller_ptr += num_equal;
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(greater_ptr),
                        compressed_greater);
                    greater_ptr += 8 - num_equal;
                }
                if (remaining != 0) {
                    unsigned const valid_mask = (1u << remaining) - 1u;
                    __m256i const v = _mm256_loadu_si256(
                        reinterpret_cast<__m256i const *>(data));
                    auto const mask_equal =
                        static_cast<unsigned>(
                            _mm256_movemask_ps(_mm256_castsi256_ps(
                                _mm256_cmpeq_epi32(pivot_vec, v)))) &
                        valid_mask;
                    auto const num_equal = _mm_popcnt_u32(mask_equal);
                    auto const mask_greater = (~mask_equal) & valid_mask;
                    auto const compressed_equal = compress_left(v, mask_equal);
                    auto const compressed_greater =
                        compress_left(v, mask_greater);
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(smaller_ptr),
                        compressed_equal);
                    smaller_ptr += num_equal;
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i *>(greater_ptr),
                        compressed_greater);
                    greater_ptr += remaining - num_equal;
                }
            }
        }
        buckets_[num_buckets_].resize(static_cast<std::size_t>(
            std::distance(buckets_[num_buckets_].data(), smaller_ptr)));
        buckets_[num_buckets_ - 1].resize(static_cast<std::size_t>(
            std::distance(buckets_[num_buckets_ - 1].data(), greater_ptr)));
        if (pivots_.size() <= num_buckets_ - 1) {
            pivots_.insert(pivots_.end(), 8, 0);
        }
        pivots_[num_buckets_ - 1] = pivot;
        ++num_buckets_;
    }

   public:
    explicit QuickHeapAVX2NoEqual() : pivots_(128), buckets_(128) {}

    void push(T e) {
        assert(buckets_.size() >= num_buckets_);
        assert((pivots_.size() & 7) == 0);
        auto const b = find_bucket(pivots_.data(), num_buckets_ - 1, e);
        assert(b < num_buckets_);
        auto &bucket = buckets_[b];
        bucket.push_back(e);
        if (bucket.capacity() < ((bucket.size() + 7u) & ~7u)) {
            auto const old_size = bucket.size();
            bucket.insert(bucket.end(), 8u - (bucket.size() & 7), 0);
            bucket.resize(old_size);
        }
        if (b == num_buckets_ - 1 && bucket.size() > 1 &&
            *(bucket.end() - 2) < e) {
            std::iter_swap(bucket.end() - 1, bucket.end() - 2);
        }
    }

    void pop() {
        assert(num_buckets_ > 0);
        assert(!buckets_[num_buckets_ - 1].empty());
        buckets_[num_buckets_ - 1].pop_back();
        if (buckets_[num_buckets_ - 1].empty() && num_buckets_ > 1) {
            --num_buckets_;
        }
        while (buckets_[num_buckets_ - 1].size() > 256 / (8 * sizeof(T))) {
            partition_last_bucket();
        }
        assert(!buckets_[num_buckets_ - 1].empty() || num_buckets_ == 1);
        if (!buckets_[num_buckets_ - 1].empty()) {
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

}  // namespace quick_heap_avx2_no_equal_detail

template <typename T>
using QuickHeapAVX2NoEqual =
    quick_heap_avx2_no_equal_detail::QuickHeapAVX2NoEqual<T>;

#undef SKIP_AVX2_POPCNT
#undef WITH_LUT
