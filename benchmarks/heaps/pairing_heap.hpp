#pragma once

#include <boost/heap/pairing_heap.hpp>

#include <functional>

template <typename T>
using pairing_heap =
    boost::heap::pairing_heap<T, boost::heap::compare<std::ranges::greater>>;
