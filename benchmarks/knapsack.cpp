#include "pq.hpp"
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

using key_type = std::int64_t;

struct Node {
    key_type upper_bound;
    std::size_t index;
    key_type free_capacity;
    key_type value;
};

constexpr unsigned slot_bits = 31;
constexpr key_type slot_mask = (key_type{1} << slot_bits) - 1;
constexpr key_type max_bound = (key_type{1} << 32) - 1;

#ifdef PQ_MONOTONE
// Children often share the upper bound of their parent, so a monotone heap must
// only see the bound
struct BoundOfKey {
    key_type operator()(key_type key) const noexcept {
        return key >> slot_bits;
    }
};
using pq_type = BENCH_PQ<key_type, BoundOfKey>;
#else
using pq_type = BENCH_PQ<key_type>;
#endif

struct Result {
    long long time_ns = 0;
    key_type solution = 0;
    long long processed_nodes = 0;
};

template <typename PQ>
Result knapsack(KnapsackInstance<key_type> const& instance, Papi& papi) {
    Result result;
    std::vector<Node> nodes;
    nodes.reserve(std::size_t{1} << 24);
    // The free slots are distinct slots of nodes, so free_slots never needs
    // more entries than nodes and grows with it. Freeing a slot then needs no
    // capacity check.
    std::vector<key_type> free_slots;
    free_slots.reserve(std::size_t{1} << 24);
    std::size_t free_slots_end = 0;
    PQ pq;
    std::size_t size = 0;
    auto push = [&](Node const& node) {
        key_type slot = 0;
        if (free_slots_end == 0) {
            slot = static_cast<key_type>(nodes.size());
            if (slot > slot_mask) [[unlikely]] {
                std::cerr << "Error: more than " << slot_mask + 1
                          << " nodes in the queue\n";
                std::exit(EXIT_FAILURE);
            }
            nodes.push_back(node);
            free_slots.resize(nodes.size());
        } else {
            slot = free_slots[--free_slots_end];
            nodes[static_cast<std::size_t>(slot)] = node;
        }
        pq.push(((max_bound - node.upper_bound) << slot_bits) | slot);
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
        pq.pop();
        --size;
        auto const slot = key & slot_mask;
        auto node = nodes[static_cast<std::size_t>(slot)];
        free_slots[free_slots_end++] = slot;
        if (node.upper_bound <= result.solution) {
            break;
        }
        auto [lb, ub] =
            instance.compute_bounds_linear(node.free_capacity, node.index + 1);
        if (node.value + lb > result.solution) {
            result.solution = node.value + lb;
        }
        if (node.index + 2 < instance.size()) {
            if (node.value + ub > result.solution) {
                push(Node{node.value + ub, node.index + 1, node.free_capacity,
                          node.value});
            }
            if (node.free_capacity >= instance.weight(node.index)) {
                node.value += instance.value(node.index);
                node.free_capacity -= instance.weight(node.index);
                ++node.index;
                push(node);
            }
        }
        ++result.processed_nodes;
    }
    auto t_end = std::chrono::steady_clock::now();
    papi.stop();
    result.time_ns = std::chrono::nanoseconds{t_end - t_start}.count();
    return result;
}

[[noreturn]] void usage(char const* name, int status) {
    auto& out = status == EXIT_SUCCESS ? std::cout : std::cerr;
    out << "Usage: " << name << " [--count-event EVENT]... INSTANCE\n"
        << "  --count-event EVENT   PAPI event counted while the search runs\n";
    std::exit(status);
}

int main(int argc, char* argv[]) {
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

    std::clog << "Reading instance...\n";
    KnapsackInstance<key_type> instance;
    try {
        instance = KnapsackInstance<key_type>(instance_file);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << instance_file.string() << ": " << e.what()
                  << '\n';
        return EXIT_FAILURE;
    }
    key_type value_sum = 0;
    for (std::size_t i = 0; i < instance.size(); ++i) {
        value_sum += instance.value(i);
    }
    if (value_sum > max_bound) {
        std::cerr << "Error: the sum of all values " << value_sum
                  << " does not fit into 32 bits\n";
        return EXIT_FAILURE;
    }
    std::clog << "Instance has " << instance.size() << " items and "
              << instance.capacity() << " capacity\n";

    std::optional<Papi> papi;
    try {
        papi.emplace(events);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    std::clog << "Working...\n";
    auto result = knapsack<pq_type>(instance, *papi);
    std::clog << "Done\n\n";

    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3)
              << static_cast<double>(result.time_ns) * 1e-9 << '\n';
    std::clog << "Solution: " << result.solution << '\n';
    std::clog << "Processed nodes: " << result.processed_nodes << '\n';
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
