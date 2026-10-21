#pragma once

#include <boost/heap/d_ary_heap.hpp>

#include <functional>

template <typename T, unsigned D>
using boost_d_ary_heap =
    boost::heap::d_ary_heap<T, boost::heap::arity<D>,
                            boost::heap::compare<std::ranges::greater>>;

template <typename T>
using boost_4_ary_heap = boost_d_ary_heap<T, 4>;

template <typename T>
using boost_8_ary_heap = boost_d_ary_heap<T, 8>;

template <typename T>
using boost_16_ary_heap = boost_d_ary_heap<T, 16>;
