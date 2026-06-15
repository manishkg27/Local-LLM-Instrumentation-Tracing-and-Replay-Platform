// =============================================================================
//  AnomalyDetector.cpp
//  -----------------------------------------------------------------------------
//  Day 2 implementation: the 4 rule stubs, the rolling-latency window for
//  LatencyHotspot, and the thread-safe AnomalyLedger.
// =============================================================================
#include "core/AnomalyDetector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fmt/format.h>

namespace llm_tui {

namespace {
// spdlog throws if you try to register the same name twice. Tests create
// many detectors in the same process, so we re-use an existing logger if
// one is already registered under "anomaly".
std::shared_ptr<spdlog::logger> make_anomaly_logger() {
    if (auto existing = spdlog::get("anomaly")) return existing;
    auto lg = spdlog::stdout_color_mt("anomaly");
    lg->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n] %v");
    lg->set_level(spdlog::level::debug);
    return lg;
}
} // namespace

// -----------------------------------------------------------------------------
//  ctor / dtor
// -----------------------------------------------------------------------------
AnomalyDetector::AnomalyDetector()
    : latency_window_(kLatencyWindow, 0.0f),
      logger_(make_anomaly_logger()) {}

// -----------------------------------------------------------------------------
//  public API
// -----------------------------------------------------------------------------
void AnomalyDetector::evaluate(const TelemetryPacket& pkt) {
    std::lock_guard<std::mutex> lk(mu_);

    // Catastrophic NaN/Inf detection
    if (pkt.kind == PacketKind::TensorStats) {
        if (std::isnan(pkt.mean) || std::isnan(pkt.max_abs) || std::isnan(pkt.sigma) ||
            std::isinf(pkt.mean) || std::isinf(pkt.max_abs) || std::isinf(pkt.sigma)) {
            record(Severity::Error, AnomalyCode::NaNInf, pkt.layer_id,
                   pkt.timestamp_ns,
                   fmt::format("layer {}: Catastrophic NaN/Inf detected in tensor activations", pkt.layer_id));
        }
        if (rule_outlier(pkt)) {
            record(Severity::Warn, AnomalyCode::OutlierFeature, pkt.layer_id,
                   pkt.timestamp_ns,
                   fmt::format("Outlier Feature Layer {}: Max > {:.1f}",
                                pkt.layer_id, cfg_.outlier_k));
        }
        if (rule_clipping(pkt)) {
            record(Severity::Error, AnomalyCode::ClippingRisk, pkt.layer_id,
                   pkt.timestamp_ns,
                   fmt::format("layer {}: |x|_max={:.3e} near fp16 max ({:.0f})",
                               pkt.layer_id, pkt.max_abs, cfg_.clipping_fp16_max));
        }
        if (rule_dead(pkt)) {
            record(Severity::Warn, AnomalyCode::DeadLayer, pkt.layer_id,
                   pkt.timestamp_ns,
                   fmt::format("layer {}: sparsity={:.2f} (threshold {:.2f})",
                               pkt.layer_id, pkt.sparsity, cfg_.dead_sparsity));
        }
    }

    // LatencyHotspot keys off LayerLatency packets.
    if (pkt.kind == PacketKind::LayerLatency) {
        push_latency_sample(pkt.latency_us);
        if (rule_hotspot(pkt)) {
            float med = median_latency_us();
            record(Severity::Warn, AnomalyCode::LatencyHotspot, pkt.layer_id,
                   pkt.timestamp_ns,
                   fmt::format("layer {}: latency={:.1f}us > {:.1f}x median({:.1f}us)",
                               pkt.layer_id, pkt.latency_us, cfg_.hotspot_factor, med));
        }
    }

    // Explicit anomaly packet evaluation
    if (pkt.kind == PacketKind::Anomaly) {
        if (pkt.anomaly_code == AnomalyCode::CpuFallback) {
            record(pkt.severity, pkt.anomaly_code, pkt.layer_id, pkt.timestamp_ns,
                   fmt::format("CUDA OOM Fallback: Processing {} on CPU Host Memory.", to_string(pkt.layer_type)));
        } else {
            record(pkt.severity, pkt.anomaly_code, pkt.layer_id, pkt.timestamp_ns,
                   "General runtime anomaly detected");
        }
    }
}

std::vector<LedgerEntry> AnomalyDetector::ledger() const {
    std::lock_guard<std::mutex> lk(mu_);
    return ledger_;
}

std::size_t AnomalyDetector::count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return ledger_.size();
}

void AnomalyDetector::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    ledger_.clear();
    latency_window_.assign(kLatencyWindow, 0.0f);
    latency_cursor_ = 0;
    latency_full_    = false;
}

// -----------------------------------------------------------------------------
//  rule stubs (tightened with proper z-scores/thresholds)
// -----------------------------------------------------------------------------
bool AnomalyDetector::rule_outlier(const TelemetryPacket& p) {
    if (p.sigma <= 0.0f) return false;          // can't z-score without a std
    return p.max_abs > cfg_.outlier_k * p.sigma;
}

bool AnomalyDetector::rule_clipping(const TelemetryPacket& p) {
    // Saturating at fp16 max is the main thing we care about.
    return p.max_abs >= cfg_.clipping_fp16_max;
}

bool AnomalyDetector::rule_dead(const TelemetryPacket& p) {
    return p.sparsity > cfg_.dead_sparsity;
}

bool AnomalyDetector::rule_hotspot(const TelemetryPacket& p) {
    if (!latency_full_ && latency_cursor_ < 8) return false;   // need a few samples
    float med = median_latency_us();
    if (med <= 0.0f) return false;
    return p.latency_us > cfg_.hotspot_factor * med;
}

// -----------------------------------------------------------------------------
//  latency rolling window
// -----------------------------------------------------------------------------
void AnomalyDetector::push_latency_sample(float us) {
    // Lock is held by caller (evaluate)
    latency_window_[latency_cursor_] = us;
    latency_cursor_ = (latency_cursor_ + 1) % kLatencyWindow;
    if (latency_cursor_ == 0) latency_full_ = true;
}

float AnomalyDetector::median_latency_us() const {
    // Lock is held by caller (evaluate / rule_hotspot)
    std::size_t n = latency_full_ ? kLatencyWindow : latency_cursor_;
    if (n == 0) return 0.0f;
    std::vector<float> tmp(latency_window_.begin(), latency_window_.begin() + n);
    std::sort(tmp.begin(), tmp.end());
    if (n % 2 == 1) return tmp[n / 2];
    return 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
}

// -----------------------------------------------------------------------------
//  ledger writer
// -----------------------------------------------------------------------------
void AnomalyDetector::record(Severity sev, AnomalyCode code,
                             std::int32_t layer_id, std::uint64_t ts,
                             std::string message) {
    // Lock is held by caller (evaluate)
    if (cfg_.log_to_stderr) {
        switch (sev) {
            case Severity::Info:  logger_->info ("[anomaly] {}", message); break;
            case Severity::Warn:  logger_->warn ("[anomaly] {}", message); break;
            case Severity::Error: logger_->error("[anomaly] {}", message); break;
        }
    }
    ledger_.push_back(LedgerEntry{sev, code, layer_id, ts, std::move(message)});
}

} // namespace llm_tui
