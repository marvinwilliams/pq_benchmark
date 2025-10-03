# Priority Queue Benchmarks

In this repository, different PQ implementations are benchmarked on various workloads.
The PQ implementations include:
- `std::priority_queue`
- `boost::heap::d_ary_heap` with different arities
- `boost::heap::fibonacci_heap`
- `boost::heap::pairing_heap`
- `multiqueue::merge_heap` The binary merge heap from the MultiQueue
- `quick_heap` A custom quick heap implementation
- `quick_heap_avx2` A AVX2 version of the quick heap.

The workloads include:
- `HeapSort`: push n elements, then pop n elements.
- `GrowShrink`: push-push-pop for n iterations, then pop-push-pop for n iterations.
- `PushPop`: Pre-fill the PQ with n elements, then perform n push-pop iterations.

Random values for the workloads are drawn uniformly at random from the range [0, n) and [0, 1000) for the `Small` variant.

Each workload besides the `HeapSort` comes in two variants:
- `Random`: the pushed values are drawn uniformly at random.
- `Monotonic`: the pushed values are `t+r` where `t` is the last popped value and `r` is drawn uniformly at random.

## Requirements:
- CMake 3.18 or higher
- A C++20 compatible compiler (tested with GCC 15.2)
- Boost
- Google Benchmark (will be downloaded automatically if not found)
- (Optional) Catch2 v3 for running tests (will be downloaded automatically if not found)

## Building
```sh
cmake --preset release
cmake --build --preset release
```

## Running Benchmarks
Running the benchmark will take several minutes. Benchmarks can be filtered using the `--benchmark_filter` flag.

```sh
./build/release/pq_benchmark --benchmark_format=csv | tee results.csv
```

## Plotting Results
Plotting requires R. The script produces a `plot.pdf` file.

```sh
Rscript scripts/plot_results.R
```
