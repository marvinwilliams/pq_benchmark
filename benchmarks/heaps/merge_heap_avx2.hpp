#pragma once

#include "competitors/merge_heap_simd.hpp"

template <typename T>
using merge_heap_avx2 = MergeHeapAVX2<T>;
