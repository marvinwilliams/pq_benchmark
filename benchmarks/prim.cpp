#include "pq.hpp"
#ifdef PQ_MONOTONE
#error Monotone pqs are not supported
#endif
#include "util/graph.hpp"
#include "util/json.hpp"
#include "util/papi.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using key_type = std::int64_t;
static constexpr key_type id_mask = (key_type{1} << 32) - 1;

struct Result {
    long long time_ns = 0;
    long long mst_weight = 0;
    long long processed_nodes = 0;
    long long ignored_nodes = 0;
};

template <typename PQ>
Result prim(Graph const& graph, std::size_t source, Papi& papi) {
    Result result;
    auto visited = std::vector<char>(graph.num_nodes());
    PQ pq;
    std::size_t size = 0;
    papi.start();
    auto t_start = std::chrono::steady_clock::now();
    visited[source] = 1;
    pq.push(static_cast<key_type>(source));
    ++size;
    while (size != 0) {
        key_type const key = pq.top();
        pq.pop();
        --size;
        auto const distance = key >> 32;
        auto const& edge = graph.edges[static_cast<std::size_t>(key & id_mask)];
        if (visited[edge.target] == 1) {
            ++result.ignored_nodes;
            continue;
        }
        result.mst_weight += edge.weight;
        for (std::size_t i = graph.nodes[edge.target];
             i < graph.nodes[edge.target + 1]; ++i) {
            auto const d = distance + graph.edges[i].weight;
            auto const target = graph.edges[i].target;
            if (visited[target] == 0) {
                pq.push((d << 32) | static_cast<key_type>(target));
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

[[noreturn]] void usage(char const* name, int status) {
    auto& out = status == EXIT_SUCCESS ? std::cout : std::cerr;
    out << "Usage: " << name
        << " [--source NODE] [--count-event EVENT]... GRAPH\n"
        << "  --source NODE         0-based index or max-degree (default 0)\n"
        << "  --count-event EVENT   PAPI event counted while the search runs\n";
    std::exit(status);
}

int main(int argc, char* argv[]) {
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

    std::clog << "Reading graph...\n";
    Graph graph;
    try {
        graph = Graph(graph_file);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << graph_file.string() << ": " << e.what()
                  << '\n';
        return EXIT_FAILURE;
    }
    std::size_t source = 0;
    try {
        source = graph.source(source_spec);
    } catch (std::logic_error const& e) {
        std::cerr << "Error: --source " << source_spec << ": " << e.what()
                  << '\n';
        return EXIT_FAILURE;
    }
    std::clog << "Graph has " << graph.num_nodes() << " nodes and "
              << graph.num_edges() << " edges, source " << source
              << " has degree " << graph.degree(source) << '\n';

    std::optional<Papi> papi;
    try {
        papi.emplace(events);
    } catch (std::runtime_error const& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    std::clog << "Working...\n";
    auto result = prim<BENCH_PQ<key_type>>(graph, source, *papi);
    std::clog << "Done\n\n";

    auto const pops = result.processed_nodes + result.ignored_nodes;
    std::clog << "= Results =\n";
    std::clog << "Time (s): " << std::fixed << std::setprecision(3)
              << static_cast<double>(result.time_ns) * 1e-9 << '\n';
    std::clog << "MST weight: " << result.mst_weight << '\n';
    std::clog << "Processed nodes: " << result.processed_nodes << '\n';
    std::clog << "Ignored nodes: " << result.ignored_nodes << '\n';
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
            obj.entry("mst_weight", result.mst_weight);
            obj.entry("processed_nodes", result.processed_nodes);
            obj.entry("ignored_nodes", result.ignored_nodes);
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
