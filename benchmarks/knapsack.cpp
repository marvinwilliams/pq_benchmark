#include "heaps.hpp"
#include "util/json.hpp"
#include "util/knapsack_instance.hpp"
#include "util/papi.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using data_type = unsigned long;

struct Node {
    data_type upper_bound;
    std::size_t index;
    data_type free_capacity;
    data_type value;
};

// Nodes live in an arena whose slots are reused. Keys pack the inverted upper
// bound into the high and the slot into the low bits, so that every min-heap,
// including the SIMD and radix heaps, works on plain integers.
using key_type = std::int64_t;
constexpr unsigned slot_bits = 31;
constexpr key_type slot_mask = (key_type{1} << slot_bits) - 1;
constexpr data_type max_bound = (data_type{1} << 32) - 1;

// Children often share the upper bound of their parent, so the radix heap must
// only see the bound to stay monotone
struct BoundOfKey {
    key_type operator()(key_type key) const noexcept {
        return key >> slot_bits;
    }
};
using radix_heap_bound = tlx::RadixHeap<key_type, BoundOfKey, key_type>;

struct Result {
    long long time_ns = 0;
    data_type solution = 0;
    long long processed_nodes = 0;
    std::size_t sum_sizes = 0;
    std::size_t max_size = 0;
};

template <typename PQ>
Result knapsack(KnapsackInstance<data_type> const& instance, Papi& papi) {
    Result result;
    std::vector<Node> nodes;
    nodes.reserve(std::size_t{1} << 24);
    std::vector<key_type> free_slots;
    PQ pq;
    std::size_t size = 0;
    auto push = [&](Node const& node) {
        key_type slot = 0;
        if (free_slots.empty()) {
            slot = static_cast<key_type>(nodes.size());
            if (slot > slot_mask) [[unlikely]] {
                std::cerr << "Error: more than " << slot_mask + 1 << " nodes in the queue\n";
                std::exit(EXIT_FAILURE);
            }
            nodes.push_back(node);
        } else {
            slot = free_slots.back();
            free_slots.pop_back();
            nodes[static_cast<std::size_t>(slot)] = node;
        }
        pq.push((static_cast<key_type>(max_bound - node.upper_bound) << slot_bits) | slot);
        ++size;
    };
    papi.start();
    auto t_start = std::chrono::steady_clock::now();
    {
        auto [lb, ub] = instance.compute_bounds_linear(instance.capacity(), 0);
        result.solution = lb;
        push(Node{ub, 0, instance.capacity(), 0});
    }
    while (size != 0) {
        key_type const key = pq.top();
        result.sum_sizes += size;
        pq.pop();
        --size;
        auto const slot = key & slot_mask;
        auto node = nodes[static_cast<std::size_t>(slot)];
        free_slots.push_back(slot);
        if (node.upper_bound <= result.solution) {
            break;
        }
        auto [lb, ub] = instance.compute_bounds_linear(node.free_capacity, node.index + 1);
        if (node.value + lb > result.solution) {
            result.solution = node.value + lb;
        }
        if (node.index + 2 < instance.size()) {
            if (node.value + ub > result.solution) {
                push(Node{node.value + ub, node.index + 1, node.free_capacity, node.value});
            }
            if (node.free_capacity >= instance.weight(node.index)) {
                node.value += instance.value(node.index);
                node.free_capacity -= instance.weight(node.index);
                ++node.index;
                push(node);
            }
        }
        result.max_size = std::max(result.max_size, size);
        ++result.processed_nodes;
    }
    auto t_end = std::chrono::steady_clock::now();
    papi.stop();
    result.time_ns = std::chrono::nanoseconds{t_end - t_start}.count();
    return result;
}

using knapsack_function = Result (*)(KnapsackInstance<data_type> const&, Papi&);

constexpr std::pair<std::string_view, knapsack_function> pqs[] = {
    {"std_pq", &knapsack<std_pq<key_type>>},
    {"mq_pq", &knapsack<mq_pq<key_type>>},
    {"boost_4_ary_heap", &knapsack<boost_4_ary_heap<key_type>>},
    {"boost_8_ary_heap", &knapsack<boost_8_ary_heap<key_type>>},
    {"boost_16_ary_heap", &knapsack<boost_16_ary_heap<key_type>>},
    {"fib_heap", &knapsack<fib_heap<key_type>>},
    {"pairing_heap", &knapsack<pairing_heap<key_type>>},
    {"merge_heap", &knapsack<merge_heap<key_type>>},
    {"s3q_queue", &knapsack<s3q_queue<key_type>>},
    {"radix_heap", &knapsack<radix_heap_bound>},
    {"quick_heap_avx2", &knapsack<quick_heap_avx2<key_type>>},
#ifdef __AVX512F__
    {"quick_heap_avx512", &knapsack<quick_heap_avx512<key_type>>},
#endif
};

[[noreturn]] void usage(char const* name, int status) {
    auto& out = status == EXIT_SUCCESS ? std::cout : std::cerr;
    out << "Usage: " << name << " --pq PQ [--count-event EVENT]... INSTANCE\n"
        << "  --pq PQ               One of";
    for (auto const& [pq_name, _] : pqs) {
        out << ' ' << pq_name;
    }
    out << "\n  --count-event EVENT   PAPI event counted while the search runs\n";
    std::exit(status);
}

int main(int argc, char* argv[]) {
    std::string pq_name;
    std::vector<std::string> events;
    std::filesystem::path instance_file;
    std::vector<std::string_view> args(argv + 1, argv + argc);
    for (std::size_t i = 0; i < args.size(); ++i) {
        auto value = [&]() -> std::string {
            if (i + 1 == args.size()) {
                std::cerr << "Error: " << args[i] << " needs a value\n";
                usage(argv[0], EXIT_FAILURE);
            }
            return std::string{args[++i]};
        };
        if (args[i] == "-h" || args[i] == "--help") {
            usage(argv[0], EXIT_SUCCESS);
        } else if (args[i] == "--pq") {
            pq_name = value();
        } else if (args[i] == "--count-event") {
            events.push_back(value());
        } else if (instance_file.empty() && !args[i].starts_with("-")) {
            instance_file = args[i];
        } else {
            std::cerr << "Error: unexpected argument " << args[i] << '\n';
            usage(argv[0], EXIT_FAILURE);
        }
    }
    if (instance_file.empty()) {
        std::cerr << "Error: no instance given\n";
        usage(argv[0], EXIT_FAILURE);
    }
    auto it = std::ranges::find(pqs, pq_name, [](auto const& p) { return p.first; });
    if (it == std::end(pqs)) {
        std::cerr << "Error: unknown pq '" << pq_name << "'\n";
        usage(argv[0], EXIT_FAILURE);
    }

    std::clog << "Reading instance...\n";
    KnapsackInstance<data_type> instance;
    try {
        instance = KnapsackInstance<data_type>(instance_file);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << instance_file.string() << ": " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    data_type value_sum = 0;
    for (std::size_t i = 0; i < instance.size(); ++i) {
        value_sum += instance.value(i);
    }
    if (value_sum > max_bound) {
        std::cerr << "Error: the sum of all values " << value_sum << " does not fit into 32 bits\n";
        return EXIT_FAILURE;
    }
    std::clog << "Instance has " << instance.size() << " items and " << instance.capacity() << " capacity\n";

    std::optional<Papi> papi;
    try {
        papi.emplace(events);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    std::clog << "Working...\n";
    auto result = it->second(instance, *papi);
    std::clog << "Done\n\n";

    auto const average_size = result.processed_nodes == 0 ? 0.0
                                                          : static_cast<double>(result.sum_sizes) /
                                                                static_cast<double>(result.processed_nodes);
    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3) << static_cast<double>(result.time_ns) * 1e-9
              << '\n';
    std::clog << "Solution: " << result.solution << '\n';
    std::clog << "Processed nodes: " << result.processed_nodes << '\n';
    std::clog << "Average PQ size: " << average_size << '\n';
    std::clog << "Max PQ size: " << result.max_size << '\n';
    for (std::size_t i = 0; i < events.size(); ++i) {
        std::clog << events[i] << ": " << papi->counters()[i] << '\n';
    }

    {
        json::Object root{std::cout};
        root.object("settings", [&](json::Object& obj) {
            obj.entry("instance_file", instance_file.string());
            obj.entry("pq", pq_name);
            obj.array("papi_events", events);
        });
        root.object("instance", [&](json::Object& obj) {
            obj.entry("num_items", instance.size());
            obj.entry("capacity", instance.capacity());
        });
        root.object("results", [&](json::Object& obj) {
            obj.entry("time_ns", result.time_ns);
            obj.entry("processed_nodes", result.processed_nodes);
            obj.entry("solution", result.solution);
            obj.entry("average_pq_size", average_size);
            obj.entry("max_pq_size", result.max_size);
            obj.object("papi", [&](json::Object& counters) {
                for (std::size_t i = 0; i < events.size(); ++i) {
                    counters.entry(events[i], papi->counters()[i]);
                }
            });
        });
    }
    std::cout << '\n';
    return EXIT_SUCCESS;
}
