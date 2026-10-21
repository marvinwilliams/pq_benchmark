#pragma once

#include <functional>
#include <queue>
#include <vector>

template <typename T>
using std_pq = std::priority_queue<T, std::vector<T>, std::ranges::greater>;
