//  AnomalyDetector.hpp
//  -----------------------------------------------------------------------------
//  Inspects TelemetryPackets and flags numerical anomalies.
//rules are real:
//         - OutlierFeature   (max(|x|) > k * sigma)
//         - ClippingRisk     (saturating at fp16/fp32 max)
//         - DeadLayer        (sparsity > 0.9 with no movement)
//         - LatencyHotspot   (latency > 3x median)
//         - CpuFallback      (CUDA OOM, ran on CPU)
//         - NaNInf           (NaN/Inf detected)
//
//  Threading model:
//      - The ledger is guarded by a std::mutex; one entry per anomaly.
//      - evaluate() and ledger() are the only public mutators / readers.
//      - They are intended to be called from the consumer thread that drains
//        the RingBuffer but are safe to call from anywhere.
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/TelemetryPacket.hpp"

namespace llm_tui {

// A single detected anomaly. Kept POD-ish (no string allocations beyond the
// message) so Panel 5 can render thousands of them without fragmenting the
// heap during a long inference run.
struct LedgerEntry {
    Severity       severity     = Severity::Info;
    AnomalyCode    code         = AnomalyCode::None;
    std::int32_t   layer_id     = -1;
    std::uint64_t  timestamp_ns = 0;
    std::string    message;         
};

class AnomalyDetector {
public:
    AnomalyDetector();

    // Inspect a single packet. Runs all enabled rules. On a hit, appends a
    // LedgerEntry to the in-memory ledger and logs via spdlog.
    void evaluate(const TelemetryPacket& pkt);

    // Read-only snapshot of the ledger (newest entries are at the back).
    std::vector<LedgerEntry> ledger() const;

    // Number of entries currently stored.
    std::size_t count() const;

    // Reset the ledger (e.g. between sessions). Tests use this.
    void clear();

    struct Config {
        float outlier_k          = 50.0f;  // |x| > k * sigma  -> OutlierFeature
                                               // RMSNorm forces sigma=1.0, and modern LLMs (SwiGLU) 
                                               // use "massive activations" up to ~50.0 for routing.
        float clipping_fp16_max  = 65504.0f;
        float clipping_fp32_max  = 3.4e38f;
        float dead_sparsity      = 0.90f;  // sparsity > dead_sparsity  -> DeadLayer
        float hotspot_factor     = 3.0f;   // latency > factor * median  -> LatencyHotspot
        bool  log_to_stderr      = true;
    };
    void set_config(const Config& c) { std::lock_guard lk(mu_); cfg_ = c; }
    Config config() const { std::lock_guard lk(mu_); return cfg_; }

private:
    // The 4 rule stubs - all return true if the rule fired.
    bool rule_outlier (const TelemetryPacket& p);
    bool rule_clipping(const TelemetryPacket& p);
    bool rule_dead    (const TelemetryPacket& p);
    bool rule_hotspot (const TelemetryPacket& p);

    // Bookkeeping for LatencyHotspot (rolling median over last 64 samples).
    static constexpr std::size_t kLatencyWindow = 64;
    std::vector<float> latency_window_;
    std::size_t        latency_cursor_ = 0;
    bool               latency_full_    = false;
    void push_latency_sample(float us);
    float median_latency_us() const;

    // Append to the ledger under the mutex.
    void record(Severity sev, AnomalyCode code, std::int32_t layer_id,
                std::uint64_t ts, std::string message);

    mutable std::mutex                     mu_;
    Config                                 cfg_;
    std::vector<LedgerEntry>               ledger_;
    std::shared_ptr<spdlog::logger>        logger_;
};

} // namespace llm_tui
