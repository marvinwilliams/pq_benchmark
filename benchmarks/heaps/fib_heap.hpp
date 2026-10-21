#pragma once

#include <boost/heap/fibonacci_heap.hpp>

#include <functional>

template <typename T>
using fib_heap =
    boost::heap::fibonacci_heap<T, boost::heap::compare<std::ranges::greater>>;
