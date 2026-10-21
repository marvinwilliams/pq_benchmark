#pragma once

#include <tlx/container/radix_heap.hpp>

#include <functional>

#define PQ_MONOTONE

template <typename T, typename KeyFn = std::identity>
using radix_heap = tlx::RadixHeap<T, KeyFn, T>;
