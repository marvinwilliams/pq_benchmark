#pragma once

#include "competitors/merge_heap.hpp"

#include <functional>

template <typename T>
using merge_heap = multiqueue::value_merge_heap<T, std::ranges::less>;
