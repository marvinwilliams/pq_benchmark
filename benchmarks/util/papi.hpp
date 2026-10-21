#pragma once

#ifdef WITH_PAPI
#include <papi.h>
#endif

#include <stdexcept>
#include <string>
#include <vector>

// Counts PAPI events of the calling thread; start/stop pairs accumulate until reset.
class Papi {
   public:
    explicit Papi(std::vector<std::string> events) : events_(std::move(events)), counters_(events_.size(), 0) {
        if (events_.empty()) {
            return;
        }
#ifdef WITH_PAPI
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
            throw std::runtime_error{"Failed to initialize PAPI library"};
        }
        if (PAPI_create_eventset(&event_set_) != PAPI_OK) {
            throw std::runtime_error{"Failed to create PAPI event set"};
        }
        for (auto const& name : events_) {
            int event = PAPI_NULL;
            if (PAPI_event_name_to_code(name.c_str(), &event) != PAPI_OK) {
                throw std::runtime_error{"PAPI event '" + name + "' not available"};
            }
            if (PAPI_add_event(event_set_, event) != PAPI_OK) {
                throw std::runtime_error{"PAPI events cannot be counted together ('" + name + "' failed)"};
            }
        }
        values_.resize(events_.size());
#else
        throw std::runtime_error{"Built without PAPI"};
#endif
    }

    Papi(Papi const&) = delete;
    Papi& operator=(Papi const&) = delete;

    ~Papi() {
#ifdef WITH_PAPI
        if (event_set_ != PAPI_NULL) {
            PAPI_cleanup_eventset(event_set_);
            PAPI_destroy_eventset(&event_set_);
        }
#endif
    }

    [[nodiscard]] bool enabled() const noexcept {
        return !events_.empty();
    }

    void start() {
#ifdef WITH_PAPI
        if (enabled() && PAPI_start(event_set_) != PAPI_OK) {
            throw std::runtime_error{"Failed to start performance counters"};
        }
#endif
    }

    void stop() {
#ifdef WITH_PAPI
        if (!enabled()) {
            return;
        }
        if (PAPI_stop(event_set_, values_.data()) != PAPI_OK) {
            throw std::runtime_error{"Failed to stop performance counters"};
        }
        for (std::size_t i = 0; i < values_.size(); ++i) {
            counters_[i] += values_[i];
        }
#endif
    }

    void reset() {
        counters_.assign(events_.size(), 0);
    }

    [[nodiscard]] std::vector<std::string> const& events() const noexcept {
        return events_;
    }

    [[nodiscard]] std::vector<long long> const& counters() const noexcept {
        return counters_;
    }

   private:
    std::vector<std::string> events_;
    std::vector<long long> counters_;
    std::vector<long long> values_;
#ifdef WITH_PAPI
    int event_set_ = PAPI_NULL;
#endif
};
