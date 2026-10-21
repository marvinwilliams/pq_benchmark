#pragma once

#include "competitors/quick_heap_simd.hpp"

template <typename T>
using quick_heap_avx2 = QuickHeapAVX2<T>;

#ifdef __AVX512F__
template <typename T>
using quick_heap_avx512 = QuickHeapAVX512<T>;
#endif
