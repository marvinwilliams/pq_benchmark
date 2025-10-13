/**
******************************************************************************
* @file:   merge_heap.hpp
*
* @author: Marvin Williams
* @date:   2021/03/02 16:21
* @brief:
*******************************************************************************
**/
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <limits>

#include <algorithm>
#include <cassert>
#include <memory>       // allocator
#include <type_traits>  // is_constructible, enable_if
#include <utility>      // move, forward
#include <vector>

namespace merge_heap_avx2_detail {

static __m256i const insert_shift_right_mask[8] = {
    _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7),
    _mm256_set_epi32(6, 5, 4, 3, 2, 1, 7, 0),
    _mm256_set_epi32(6, 5, 4, 3, 2, 7, 1, 0),
    _mm256_set_epi32(6, 5, 4, 3, 7, 2, 1, 0),
    _mm256_set_epi32(6, 5, 4, 7, 3, 2, 1, 0),
    _mm256_set_epi32(6, 5, 7, 4, 3, 2, 1, 0),
    _mm256_set_epi32(6, 7, 5, 4, 3, 2, 1, 0),
    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
};

static __m256i const insert_shift_left_mask[8] = {
    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
    _mm256_set_epi32(7, 6, 5, 4, 3, 2, 0, 1),
    _mm256_set_epi32(7, 6, 5, 4, 3, 0, 2, 1),
    _mm256_set_epi32(7, 6, 5, 4, 0, 3, 2, 1),
    _mm256_set_epi32(7, 6, 5, 0, 4, 3, 2, 1),
    _mm256_set_epi32(7, 6, 0, 5, 4, 3, 2, 1),
    _mm256_set_epi32(7, 0, 6, 5, 4, 3, 2, 1),
    _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1),
};

static __m256i const max_epi32 =
    _mm256_set1_epi32(std::numeric_limits<std::int32_t>::max());

static __m256i const indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

inline void inplace_merge(__m256i a, __m256i b, std::int32_t *out_lo,
                          std::int32_t *out_hi) {
    __m256i const b_rev = _mm256_permutevar8x32_epi32(
        b, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256i merged_lo = _mm256_min_epi32(a, b_rev);
    __m256i merged_hi = _mm256_max_epi32(a, b_rev);

    __m256i shuf_lo = _mm256_permutevar8x32_epi32(
        merged_lo, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
    __m256i shuf_hi = _mm256_permutevar8x32_epi32(
        merged_hi, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
    __m256i min_lo = _mm256_min_epi32(merged_lo, shuf_lo);
    __m256i max_lo = _mm256_max_epi32(merged_lo, shuf_lo);
    __m256i min_hi = _mm256_min_epi32(merged_hi, shuf_hi);
    __m256i max_hi = _mm256_max_epi32(merged_hi, shuf_hi);
    merged_lo = _mm256_blend_epi32(min_lo, max_lo, 0b11110000);
    merged_hi = _mm256_blend_epi32(min_hi, max_hi, 0b11110000);

    shuf_lo = _mm256_shuffle_epi32(merged_lo, _MM_SHUFFLE(1, 0, 3, 2));
    shuf_hi = _mm256_shuffle_epi32(merged_hi, _MM_SHUFFLE(1, 0, 3, 2));
    min_lo = _mm256_min_epi32(merged_lo, shuf_lo);
    max_lo = _mm256_max_epi32(merged_lo, shuf_lo);
    min_hi = _mm256_min_epi32(merged_hi, shuf_hi);
    max_hi = _mm256_max_epi32(merged_hi, shuf_hi);
    merged_lo = _mm256_blend_epi32(min_lo, max_lo, 0b11001100);
    merged_hi = _mm256_blend_epi32(min_hi, max_hi, 0b11001100);

    shuf_lo = _mm256_shuffle_epi32(merged_lo, _MM_SHUFFLE(2, 3, 0, 1));
    shuf_hi = _mm256_shuffle_epi32(merged_hi, _MM_SHUFFLE(2, 3, 0, 1));
    min_lo = _mm256_min_epi32(merged_lo, shuf_lo);
    max_lo = _mm256_max_epi32(merged_lo, shuf_lo);
    min_hi = _mm256_min_epi32(merged_hi, shuf_hi);
    max_hi = _mm256_max_epi32(merged_hi, shuf_hi);
    merged_lo = _mm256_blend_epi32(min_lo, max_lo, 0b10101010);
    merged_hi = _mm256_blend_epi32(min_hi, max_hi, 0b10101010);

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_lo), merged_lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_hi), merged_hi);
}

inline void inplace_merge(std::int32_t *in1, std::int32_t *in2,
                          std::int32_t *out_lo, std::int32_t *out_hi) {
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
    static constexpr std::size_t NodeSize = 256 / (8 * sizeof(T));
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
        __m256i const value_vec = _mm256_set1_epi32(value);
        __m256i const buffer_vec = _mm256_loadu_si256(
            reinterpret_cast<__m256i const *>(push_buffer_.data()));
        int const num_smaller = _mm_popcnt_u32(
            static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                _mm256_cmpgt_epi32(value_vec, buffer_vec)))) &
            ((1u << static_cast<unsigned>(push_buffer_size_)) - 1u));
        __m256i const sorted = _mm256_permutevar8x32_epi32(
            buffer_vec, insert_shift_right_mask[num_smaller]);
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
            size_type const old_size = size();
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
            (pop_buffer_size_ < 8 &&
             (data_.empty() || value <= data_.front().back()))) {
            unsigned const valid_mask =
                (1u << (8u - static_cast<unsigned>(pop_buffer_size_))) - 1u;
            __m256i const value_vec = _mm256_set1_epi32(value);
            __m256i const data_vec = _mm256_loadu_si256(
                reinterpret_cast<__m256i const *>(pop_buffer_.data()));
            int const num_smaller = _mm_popcnt_u32(
                static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                    _mm256_cmpgt_epi32(value_vec, data_vec)))) |
                valid_mask);
            if (pop_buffer_size_ == NodeSize) {
                assert(num_smaller < 8);
                insert_push_buffer(pop_buffer_.back());
                __m256i const sorted = _mm256_permutevar8x32_epi32(
                    _mm256_blend_epi32(data_vec, value_vec, 0x80),
                    insert_shift_right_mask[num_smaller]);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(pop_buffer_.data()), sorted);
            } else {
                assert(num_smaller > 0 && num_smaller <= 8);
                __m256i const sorted = _mm256_permutevar8x32_epi32(
                    _mm256_blend_epi32(data_vec, value_vec, 0x1),
                    insert_shift_left_mask[num_smaller - 1]);
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
                    _mm256_set1_epi32(static_cast<int>(push_buffer_size_) - 1);
                __m256i const mask = _mm256_cmpgt_epi32(indices, size_vec);
                __m256i const blended =
                    _mm256_blendv_epi8(push_buffer_vec, max_epi32, mask);
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

// template <typename Key, typename T, typename Comparator =
// std::less<Key>, std::size_t NodeSize = 64,
//           typename Allocator = std::allocator<std::pair<Key, T>>>
// using key_value_merge_heap =
//     merge_heap<std::pair<Key, T>, Key,
//     util::get_nth<std::pair<Key, T>, 0>, Comparator, NodeSize,
//     Allocator>;

}  // namespace merge_heap_avx2_detail

template <typename T, typename Allocator = std::allocator<T>>
using MergeHeapAVX2 = merge_heap_avx2_detail::MergeHeapAVX2<T, Allocator>;
