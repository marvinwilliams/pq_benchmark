#pragma once

/**
******************************************************************************
* @file:   merge_heap.hpp
*
* @author: Marvin Williams
* @date:   2021/03/02 16:21
* @brief:
*******************************************************************************
**/

#ifndef __AVX2__
#error "merge_heap_simd.hpp requires AVX2 (e.g. -march=native)"
#endif

#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>       // allocator
#include <type_traits>  // is_constructible, enable_if
#include <utility>      // move, forward
#include <vector>

namespace merge_heap_avx2_detail {

template <typename T>
struct Simd;

template <>
struct Simd<std::int32_t> {
    static constexpr std::size_t lanes = 8;

    static __m256i set1(std::int32_t value) {
        return _mm256_set1_epi32(value);
    }

    static __m256i lane_indices() {
        return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    }

    static __m256i cmpgt(__m256i a, __m256i b) {
        return _mm256_cmpgt_epi32(a, b);
    }

    static unsigned movemask(__m256i mask) {
        return static_cast<unsigned>(
            _mm256_movemask_ps(_mm256_castsi256_ps(mask)));
    }

    static void minmax(__m256i a, __m256i b, __m256i &min, __m256i &max) {
        min = _mm256_min_epi32(a, b);
        max = _mm256_max_epi32(a, b);
    }

    static __m256i reverse(__m256i v) {
        return _mm256_permutevar8x32_epi32(
            v, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    }

    // Swaps every lane i with lane i ^ D
    template <std::size_t D>
    static __m256i swap_lanes(__m256i v) {
        if constexpr (D == 4) {
            return _mm256_permutevar8x32_epi32(
                v, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
        } else if constexpr (D == 2) {
            return _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
        } else {
            static_assert(D == 1);
            return _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
        }
    }
};

template <>
struct Simd<std::int64_t> {
    static constexpr std::size_t lanes = 4;

    static __m256i set1(std::int64_t value) {
        return _mm256_set1_epi64x(value);
    }

    static __m256i lane_indices() { return _mm256_set_epi64x(3, 2, 1, 0); }

    static __m256i cmpgt(__m256i a, __m256i b) {
        return _mm256_cmpgt_epi64(a, b);
    }

    static unsigned movemask(__m256i mask) {
        return static_cast<unsigned>(
            _mm256_movemask_pd(_mm256_castsi256_pd(mask)));
    }

    static void minmax(__m256i a, __m256i b, __m256i &min, __m256i &max) {
        __m256i const gt = _mm256_cmpgt_epi64(a, b);
        min = _mm256_blendv_epi8(a, b, gt);
        max = _mm256_blendv_epi8(b, a, gt);
    }

    static __m256i reverse(__m256i v) {
        return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(0, 1, 2, 3));
    }

    // Swaps every lane i with lane i ^ D
    template <std::size_t D>
    static __m256i swap_lanes(__m256i v) {
        if constexpr (D == 2) {
            return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(1, 0, 3, 2));
        } else {
            static_assert(D == 1);
            return _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
        }
    }
};

// Immediate for _mm256_blend_epi32 that takes the key lanes i with i & D set
// from the second operand
template <std::size_t Lanes, std::size_t D>
constexpr int blend_lanes_with_bit() {
    constexpr std::size_t width = 8 / Lanes;
    int imm = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        if ((i / width) & D) {
            imm |= 1 << i;
        }
    }
    return imm;
}

// Immediate for _mm256_blend_epi32 that takes the key lane `Lane` from the
// second operand
template <std::size_t Lanes, std::size_t Lane>
constexpr int blend_lane() {
    constexpr std::size_t width = 8 / Lanes;
    return ((1 << width) - 1) << (Lane * width);
}

using permutation = std::array<std::int32_t, 8>;

// Expands a permutation of key lanes to one of 32-bit lanes for
// _mm256_permutevar8x32_epi32
template <std::size_t Lanes>
constexpr permutation expand(std::array<std::size_t, Lanes> const &lanes) {
    constexpr std::size_t width = 8 / Lanes;
    permutation result{};
    for (std::size_t i = 0; i < 8; ++i) {
        result[i] = static_cast<std::int32_t>(lanes[i / width] * width +
                                              i % width);
    }
    return result;
}

// insert_shift_right[k] moves the last lane to lane k and the lanes k, k + 1,
// ... one to the right
template <std::size_t Lanes>
constexpr std::array<permutation, Lanes> make_insert_shift_right() {
    std::array<permutation, Lanes> result{};
    for (std::size_t k = 0; k < Lanes; ++k) {
        std::array<std::size_t, Lanes> lanes{};
        for (std::size_t j = 0; j < Lanes; ++j) {
            lanes[j] = j < k ? j : j == k ? Lanes - 1 : j - 1;
        }
        result[k] = expand(lanes);
    }
    return result;
}

// insert_shift_left[k] moves the first lane to lane k and the lanes 1, ..., k
// one to the left
template <std::size_t Lanes>
constexpr std::array<permutation, Lanes> make_insert_shift_left() {
    std::array<permutation, Lanes> result{};
    for (std::size_t k = 0; k < Lanes; ++k) {
        std::array<std::size_t, Lanes> lanes{};
        for (std::size_t j = 0; j < Lanes; ++j) {
            lanes[j] = j < k ? j + 1 : j == k ? 0 : j;
        }
        result[k] = expand(lanes);
    }
    return result;
}

template <std::size_t Lanes>
inline constexpr auto insert_shift_right = make_insert_shift_right<Lanes>();

template <std::size_t Lanes>
inline constexpr auto insert_shift_left = make_insert_shift_left<Lanes>();

inline __m256i permute(__m256i v, permutation const &p) {
    return _mm256_permutevar8x32_epi32(
        v, _mm256_loadu_si256(reinterpret_cast<__m256i const *>(p.data())));
}

// One step of the bitonic merge network, comparing lanes i and i ^ D
template <typename T, std::size_t D>
inline void bitonic_step(__m256i &lo, __m256i &hi) {
    using S = Simd<T>;
    constexpr int imm = blend_lanes_with_bit<S::lanes, D>();
    __m256i min_lo, max_lo, min_hi, max_hi;
    S::minmax(lo, S::template swap_lanes<D>(lo), min_lo, max_lo);
    S::minmax(hi, S::template swap_lanes<D>(hi), min_hi, max_hi);
    lo = _mm256_blend_epi32(min_lo, max_lo, imm);
    hi = _mm256_blend_epi32(min_hi, max_hi, imm);
    if constexpr (D > 1) {
        bitonic_step<T, D / 2>(lo, hi);
    }
}

template <typename T>
inline void inplace_merge(__m256i a, __m256i b, T *out_lo, T *out_hi) {
    using S = Simd<T>;
    __m256i lo, hi;
    S::minmax(a, S::reverse(b), lo, hi);
    bitonic_step<T, S::lanes / 2>(lo, hi);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_lo), lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_hi), hi);
}

template <typename T>
inline void inplace_merge(T *in1, T *in2, T *out_lo, T *out_hi) {
    inplace_merge(_mm256_loadu_si256(reinterpret_cast<__m256i const *>(in1)),
                  _mm256_loadu_si256(reinterpret_cast<__m256i const *>(in2)),
                  out_lo, out_hi);
}

template <typename T, typename Allocator = std::allocator<T>>
class MergeHeapAVX2 {
    static_assert(std::is_same_v<T, std::int32_t> ||
                  std::is_same_v<T, std::int64_t>);

   public:
    using value_type = T;
    using reference = T &;
    using const_reference = T const &;

   private:
    using S = Simd<T>;
    static constexpr std::size_t NodeSize = S::lanes;
    using node_type = std::array<value_type, NodeSize>;
    using allocator_type = Allocator;
    using container_type = std::vector<node_type>;
    using iterator = typename container_type::const_iterator;
    using const_iterator = typename container_type::const_iterator;
    using difference_type = typename container_type::difference_type;
    using size_type = std::size_t;

   private:
    node_type push_buffer_;
    node_type pop_buffer_;
    std::size_t push_buffer_size_ = 0;
    std::size_t pop_buffer_size_ = 0;
    container_type data_;

   private:
    static constexpr size_type parent_index(size_type const index) noexcept {
        assert(index > 0);
        return (index - 1) >> 1;
    }

    static constexpr size_type first_child_index(
        size_type const index) noexcept {
        return (index << 1) + 1;
    }

    constexpr bool compare_last(size_type const lhs,
                                size_type const rhs) const noexcept {
        return data_[lhs].back() < data_[rhs].back();
    }

    size_type min_child_index(size_type index) const
        noexcept(noexcept(compare_last(0, 0))) {
        assert(index < data_.size());
        index = first_child_index(index);
        assert(index + 1 < data_.size());
        return index + static_cast<size_type>(compare_last(index + 1, index));
    }

    void insert_push_buffer(value_type const &value) {
        assert(push_buffer_size_ < NodeSize);
        push_buffer_.back() = value;
        __m256i const value_vec = S::set1(value);
        __m256i const buffer_vec = _mm256_loadu_si256(
            reinterpret_cast<__m256i const *>(push_buffer_.data()));
        auto const num_smaller = static_cast<std::size_t>(_mm_popcnt_u32(
            S::movemask(S::cmpgt(value_vec, buffer_vec)) &
            ((1u << static_cast<unsigned>(push_buffer_size_)) - 1u)));
        __m256i const sorted =
            permute(buffer_vec, insert_shift_right<NodeSize>[num_smaller]);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(push_buffer_.data()),
                            sorted);
        ++push_buffer_size_;
        if (push_buffer_size_ == NodeSize) {
            insert(push_buffer_.begin(), push_buffer_.end());
            push_buffer_size_ = 0;
        }
    }

   public:
    MergeHeapAVX2() = default;

    explicit MergeHeapAVX2(allocator_type const &alloc) : data_(alloc) {}

    inline iterator begin() const noexcept { return data_.cbegin(); }

    inline const_iterator cbegin() const noexcept { return data_.cbegin(); }

    inline iterator end() const noexcept { return data_.cend(); }

    inline const_iterator cend() const noexcept { return data_.cend(); }

    [[nodiscard]] inline bool empty() const noexcept {
        return pop_buffer_size_ == 0;
    }

    inline size_type size() const noexcept {
        return data_.size() * NodeSize + pop_buffer_size_ + push_buffer_size_;
    }

    inline const_reference top() const {
        assert(!empty());
        return *(pop_buffer_.end() - pop_buffer_size_);
    }

    inline node_type const &top_node() const {
        assert(!empty());
        return data_.front();
    }

    inline node_type &top_node() {
        assert(!empty());
        return data_.front();
    }

    inline void reserve(std::size_t const cap) {
        data_.reserve(cap / NodeSize + (cap % NodeSize == 0 ? 0 : 1));
    }

    inline void reserve_and_touch(std::size_t const cap) {
        auto const num_nodes = cap / NodeSize + (cap % NodeSize == 0 ? 0 : 1);
        if (data_.size() < num_nodes) {
            auto const old_size = data_.size();
            data_.resize(num_nodes);
            // this does not free allocated memory
            data_.resize(old_size);
        }
    }

    void pop_node() {
        assert(!empty());
        size_type index = 0;
        size_type const first_incomplete_parent = parent_index(data_.size());
        while (index < first_incomplete_parent) {
            auto min_child = first_child_index(index);
            auto max_child = min_child + 1;
            assert(max_child < data_.size());
            if (compare_last(max_child, min_child)) {
                std::swap(min_child, max_child);
            }
            inplace_merge(data_[min_child].data(), data_[max_child].data(),
                          data_[index].data(), data_[max_child].data());
            index = min_child;
        }
        // If we have a child, we cannot have two, so we can just move the node
        // into the hole.
        if (first_child_index(index) + 1 == data_.size()) {
            std::move(data_.back().begin(), data_.back().end(),
                      data_[index].begin());
        } else if (index + 1 < data_.size()) {
            size_type parent;
            while (index > 0 && (parent = parent_index(index),
                                 data_.back().front() < data_[parent].back())) {
                inplace_merge(data_[parent].data(), data_.back().data(),
                              data_.back().data(), data_[index].data());
                index = parent;
            }
            std::move(data_.back().begin(), data_.back().end(),
                      data_[index].begin());
        }
        data_.pop_back();
    }

    template <typename Iter>
    void extract_top_node(Iter output) {
        assert(!empty());
        std::move(data_.front().begin(), data_.front().end(), output);
        pop_node();
    }

    void push(value_type const &value) {
        if (pop_buffer_size_ == 0 || value <= pop_buffer_.back() ||
            (pop_buffer_size_ < NodeSize &&
             (data_.empty() || value <= data_.front().front()) &&
             (push_buffer_size_ == 0 || value <= push_buffer_.front()))) {
            unsigned const valid_mask =
                (1u << (NodeSize - pop_buffer_size_)) - 1u;
            __m256i const value_vec = S::set1(value);
            __m256i const data_vec = _mm256_loadu_si256(
                reinterpret_cast<__m256i const *>(pop_buffer_.data()));
            auto const num_smaller = static_cast<std::size_t>(_mm_popcnt_u32(
                S::movemask(S::cmpgt(value_vec, data_vec)) | valid_mask));
            if (pop_buffer_size_ == NodeSize) {
                assert(num_smaller < NodeSize);
                insert_push_buffer(pop_buffer_.back());
                __m256i const sorted = permute(
                    _mm256_blend_epi32(data_vec, value_vec,
                                       blend_lane<NodeSize, NodeSize - 1>()),
                    insert_shift_right<NodeSize>[num_smaller]);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(pop_buffer_.data()), sorted);
            } else {
                assert(num_smaller > 0 && num_smaller <= NodeSize);
                __m256i const sorted = permute(
                    _mm256_blend_epi32(data_vec, value_vec,
                                       blend_lane<NodeSize, 0>()),
                    insert_shift_left<NodeSize>[num_smaller - 1]);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(pop_buffer_.data()), sorted);
                ++pop_buffer_size_;
            }
            return;
        }
        insert_push_buffer(value);
    }

    void pop() {
        assert(pop_buffer_size_ > 0);
        --pop_buffer_size_;
        if (pop_buffer_size_ == 0) {
            if (data_.empty()) {
                if (push_buffer_size_ > 0) {
                    std::move(push_buffer_.begin(),
                              push_buffer_.begin() + push_buffer_size_,
                              pop_buffer_.end() - push_buffer_size_);
                    pop_buffer_size_ = push_buffer_size_;
                    push_buffer_size_ = 0;
                }
                return;
            }
            if (push_buffer_size_ == 0) {
                __m256i const data = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(data_.front().data()));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(pop_buffer_.data()), data);
            } else {
                __m256i const push_buffer_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(push_buffer_.data()));
                __m256i const size_vec =
                    S::set1(static_cast<T>(push_buffer_size_) - 1);
                __m256i const mask = S::cmpgt(S::lane_indices(), size_vec);
                __m256i const blended = _mm256_blendv_epi8(
                    push_buffer_vec, S::set1(std::numeric_limits<T>::max()),
                    mask);
                inplace_merge(
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(
                        data_.front().data())),
                    blended, pop_buffer_.data(), push_buffer_.data());
            }
            pop_buffer_size_ = NodeSize;
            pop_node();
        }
    }

    template <typename Iter>
    void insert(Iter first, Iter last) {
        auto index = data_.size();
        data_.push_back({});
        while (index > 0) {
            auto const parent = parent_index(index);
            assert(parent < index);
            if (*first >= data_[parent].back()) {
                break;
            }
            inplace_merge(data_[parent].data(), &(*first), &(*first),
                          data_[index].data());
            index = parent;
        }
        std::move(first, last, data_[index].begin());
    }

    inline void clear() noexcept {
        pop_buffer_size_ = 0;
        push_buffer_size_ = 0;
        data_.clear();
    }
};

}  // namespace merge_heap_avx2_detail

template <typename T, typename Allocator = std::allocator<T>>
using MergeHeapAVX2 = merge_heap_avx2_detail::MergeHeapAVX2<T, Allocator>;
