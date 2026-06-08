// =============================================================================
//  AnomalyDetector.hpp
//  -----------------------------------------------------------------------------
//  Inspects TelemetryPackets and flags numerical anomalies.
//
//  Day 1: STUB — prints "no anomalies yet" on a single packet.
//  Day 5: full implementation with rules:
//        - OutlierFeature   (max(|x|) > k * sigma)
//        - ClippingRisk     (saturating at fp16/fp32 max)
//        - DeadLayer        (sparsity > 0.9 with no movement)
//        - LatencyHotspot   (latency > 3x median)
//        - CpuFallback      (CUDA OOM, ran on CPU)
//        - NaNInf           (NaN/Inf detected)
// =============================================================================
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <memory>
#include <string>
#include <vector>

#include "core/TelemetryPacket.hpp"

namespace llm_tui {

class AnomalyDetector {
public:
    AnomalyDetector() : logger_(spdlog::stdout_color_mt("anomaly")) {
        logger_->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    }

    // Day 1: just log; Day 5: actually flag and return modified packets.
    void inspect(const TelemetryPacket& pkt) {
        if (pkt.kind == PacketKind::TensorStats) {
            logger_->debug("inspect shape=[{},{},{},{}] max={:.3f} sigma={:.3f} sparsity={:.2f}%%",
                           pkt.shape[0], pkt.shape[1], pkt.shape[2], pkt.shape[3],
                           pkt.max_abs, pkt.sigma, pkt.sparsity);
        }
    }

    // Day 5: returns the count of detected anomalies so far.
    std::size_t count() const { return 0; }

private:
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace llm_tui
