#include "heaps.hpp"
#include "util/graph.hpp"
#include "util/json.hpp"
#include "util/papi.hpp"


#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Keys pack the distance into the high and the node into the low bits, so that
// every heap, including the SIMD and radix heaps, works on plain integers.
using key_type = std::int64_t;

struct Result {
    long long time_ns = 0;
    std::vector<long long> distances;
    long long processed_nodes = 0;
    long long ignored_nodes = 0;
    std::size_t sum_sizes = 0;
    std::size_t max_size = 0;
};

template <typename PQ>
Result dijkstra(Graph const& graph, std::size_t source, Papi& papi) {
    auto const id_bits = static_cast<unsigned>(std::bit_width(graph.num_nodes()));
    auto const id_mask = (key_type{1} << id_bits) - 1;
    auto const max_distance = std::numeric_limits<key_type>::max() >> id_bits;
    Result result;
    result.distances.assign(graph.num_nodes(), std::numeric_limits<long long>::max());
    PQ pq;
    std::size_t size = 0;
    papi.start();
    auto t_start = std::chrono::steady_clock::now();
    result.distances[source] = 0;
    pq.push(static_cast<key_type>(source));
    ++size;
    while (size != 0) {
        result.sum_sizes += size;
        result.max_size = std::max(result.max_size, size);
        key_type const key = pq.top();
        pq.pop();
        --size;
        auto const distance = key >> id_bits;
        auto const node = static_cast<std::size_t>(key & id_mask);
        // Ignore stale nodes
        if (distance > result.distances[node]) {
            ++result.ignored_nodes;
            continue;
        }
        for (std::size_t i = graph.nodes[node]; i < graph.nodes[node + 1]; ++i) {
            auto const d = distance + graph.edges[i].weight;
            auto const target = graph.edges[i].target;
            if (d < result.distances[target]) {
                if (d > max_distance) [[unlikely]] {
                    std::cerr << "Error: distance " << d << " does not fit into " << 63 - id_bits << " bits\n";
                    std::exit(EXIT_FAILURE);
                }
                result.distances[target] = d;
                pq.push((d << id_bits) | static_cast<key_type>(target));
                ++size;
            }
        }
        ++result.processed_nodes;
    }
    auto t_end = std::chrono::steady_clock::now();
    papi.stop();
    result.time_ns = std::chrono::nanoseconds{t_end - t_start}.count();
    return result;
}

using dijkstra_function = Result (*)(Graph const&, std::size_t, Papi&);

constexpr std::pair<std::string_view, dijkstra_function> pqs[] = {
    {"std_pq", &dijkstra<std_pq<key_type>>},
    {"mq_pq", &dijkstra<mq_pq<key_type>>},
    {"boost_4_ary_heap", &dijkstra<boost_4_ary_heap<key_type>>},
    {"boost_8_ary_heap", &dijkstra<boost_8_ary_heap<key_type>>},
    {"boost_16_ary_heap", &dijkstra<boost_16_ary_heap<key_type>>},
    {"fib_heap", &dijkstra<fib_heap<key_type>>},
    {"pairing_heap", &dijkstra<pairing_heap<key_type>>},
    {"merge_heap", &dijkstra<merge_heap<key_type>>},
    {"s3q_queue", &dijkstra<s3q_queue<key_type>>},
    {"radix_heap", &dijkstra<radix_heap<key_type>>},
    {"quick_heap_avx2", &dijkstra<quick_heap_avx2<key_type>>},
#ifdef __AVX512F__
    {"quick_heap_avx512", &dijkstra<quick_heap_avx512<key_type>>},
#endif
};

bool has_zero_weight_edge(Graph const& graph) {
    for (std::size_t u = 0; u < graph.num_nodes(); ++u) {
        for (std::size_t i = graph.nodes[u]; i < graph.nodes[u + 1]; ++i) {
            if (graph.edges[i].weight <= 0 && graph.edges[i].target != u) {
                return true;
            }
        }
    }
    return false;
}

[[noreturn]] void usage(char const* name, int status) {
    auto& out = status == EXIT_SUCCESS ? std::cout : std::cerr;
    out << "Usage: " << name << " --pq PQ [--source NODE] [--count-event EVENT]... GRAPH\n"
        << "  --pq PQ               One of";
    for (auto const& [pq_name, _] : pqs) {
        out << ' ' << pq_name;
    }
    out << "\n  --source NODE         0-based index or max-degree (default 0)\n"
        << "  --count-event EVENT   PAPI event counted while the search runs\n";
    std::exit(status);
}

int main(int argc, char* argv[]) {
    std::string pq_name;
    std::string source_spec = "0";
    std::vector<std::string> events;
    std::filesystem::path graph_file;
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
        } else if (args[i] == "--source") {
            source_spec = value();
        } else if (args[i] == "--count-event") {
            events.push_back(value());
        } else if (graph_file.empty() && !args[i].starts_with("-")) {
            graph_file = args[i];
        } else {
            std::cerr << "Error: unexpected argument " << args[i] << '\n';
            usage(argv[0], EXIT_FAILURE);
        }
    }
    if (graph_file.empty()) {
        std::cerr << "Error: no graph given\n";
        usage(argv[0], EXIT_FAILURE);
    }
    auto it = std::ranges::find(pqs, pq_name, [](auto const& p) { return p.first; });
    if (it == std::end(pqs)) {
        std::cerr << "Error: unknown pq '" << pq_name << "'\n";
        usage(argv[0], EXIT_FAILURE);
    }

    std::clog << "Reading graph...\n";
    Graph graph;
    try {
        graph = Graph(graph_file);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << graph_file.string() << ": " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    std::size_t source = 0;
    try {
        source = graph.source(source_spec);
    } catch (std::logic_error const& e) {
        std::cerr << "Error: --source " << source_spec << ": " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    // A zero weight edge can push a key below the last popped one, which the radix heap does not allow
    if (pq_name == "radix_heap" && has_zero_weight_edge(graph)) {
        std::cerr << "Error: radix_heap needs positive edge weights\n";
        return EXIT_FAILURE;
    }
    std::clog << "Graph has " << graph.num_nodes() << " nodes and " << graph.num_edges() << " edges, source " << source
              << " has degree " << graph.degree(source) << '\n';

    std::optional<Papi> papi;
    try {
        papi.emplace(events);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    std::clog << "Working...\n";
    auto result = it->second(graph, source, *papi);
    std::clog << "Done\n\n";

    auto furthest_node = std::ranges::max_element(result.distances, [](auto const& a, auto const& b) {
        if (b == std::numeric_limits<long long>::max()) {
            return false;
        }
        if (a == std::numeric_limits<long long>::max()) {
            return true;
        }
        return a < b;
    });
    auto const pops = result.processed_nodes + result.ignored_nodes;
    auto const average_size = static_cast<double>(result.sum_sizes) / static_cast<double>(pops);
    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3) << static_cast<double>(result.time_ns) * 1e-9
              << '\n';
    std::clog << "Furthest node: " << furthest_node - result.distances.begin() << '\n';
    std::clog << "Longest distance: " << *furthest_node << '\n';
    std::clog << "Processed nodes: " << result.processed_nodes << '\n';
    std::clog << "Ignored nodes: " << result.ignored_nodes << '\n';
    std::clog << "Average PQ size: " << average_size << '\n';
    std::clog << "Max PQ size: " << result.max_size << '\n';
    for (std::size_t i = 0; i < events.size(); ++i) {
        std::clog << events[i] << ": " << papi->counters()[i] << '\n';
    }

    {
        json::Object root{std::cout};
        root.object("settings", [&](json::Object& obj) {
            obj.entry("graph_file", graph_file.string());
            obj.entry("source", source_spec);
            obj.entry("pq", pq_name);
            obj.array("papi_events", events);
        });
        root.object("graph", [&](json::Object& obj) {
            obj.entry("num_nodes", graph.num_nodes());
            obj.entry("num_edges", graph.num_edges());
            obj.entry("source", source);
            obj.entry("source_degree", graph.degree(source));
        });
        root.object("results", [&](json::Object& obj) {
            obj.entry("time_ns", result.time_ns);
            obj.entry("furthest_node", furthest_node - result.distances.begin());
            obj.entry("longest_distance", *furthest_node);
            obj.entry("processed_nodes", result.processed_nodes);
            obj.entry("ignored_nodes", result.ignored_nodes);
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
