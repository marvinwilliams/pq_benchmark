#pragma once

#include "competitors/mq_heap.hpp"

#include <functional>

template <typename T>
using mq_pq = multiqueue::Heap<T, std::ranges::greater>;
