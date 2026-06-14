// =============================================================================
//  test_anomaly.cpp
//  -----------------------------------------------------------------------------
//  Catch2 v3 tests for the Day-2 AnomalyDetector rule stubs and ledger.
// =============================================================================
#include <catch2/catch_test_macros.hpp>

#include "core/AnomalyDetector.hpp"
#include "core/TelemetryPacket.hpp"

using namespace llm_tui;

namespace {
TelemetryPacket make_tensor(std::int32_t layer_id,
                            float max_abs,
                            float sigma,
                            float sparsity) {
    auto p = make_packet(PacketKind::TensorStats);
    p.layer_id = layer_id;
    p.max_abs  = max_abs;
    p.sigma    = sigma;
    p.sparsity = sparsity;
    return p;
}
TelemetryPacket make_latency(std::int32_t layer_id, float us) {
    auto p = make_packet(PacketKind::LayerLatency);
    p.layer_id   = layer_id;
    p.latency_us = us;
    return p;
}
} // namespace

TEST_CASE("AnomalyDetector starts empty", "[anomaly]") {
    AnomalyDetector d;
    REQUIRE(d.count() == 0);
    REQUIRE(d.ledger().empty());
}

TEST_CASE("AnomalyDetector fires OutlierFeature when |x|>k*sigma", "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.log_to_stderr = false});
    // Default k=50.0, sigma=1.0 -> threshold 50.0.  Use 60.0 to trigger Outlier
    // only (max_abs=60 << 65504 fp16 clipping limit, sparsity=0.1 < 0.9 dead).
    d.evaluate(make_tensor(3, /*max=*/60.0f, /*sigma=*/1.0f, /*sparsity=*/0.1f));
    REQUIRE(d.count() == 1);
    auto lg = d.ledger();
    REQUIRE(lg[0].code == AnomalyCode::OutlierFeature);
    REQUIRE(lg[0].layer_id == 3);
    REQUIRE(lg[0].severity == Severity::Warn);
}

TEST_CASE("AnomalyDetector does NOT fire OutlierFeature when sigma=0", "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.log_to_stderr = false});
    d.evaluate(make_tensor(0, /*max=*/100.0f, /*sigma=*/0.0f, /*sparsity=*/0.1f));
    REQUIRE(d.count() == 0);
}

TEST_CASE("AnomalyDetector fires ClippingRisk near fp16 max", "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.log_to_stderr = false});
    // Use small sigma so Outlier doesn't also fire.
    d.evaluate(make_tensor(7, /*max=*/65504.0f, /*sigma=*/100000.0f, /*sparsity=*/0.0f));
    REQUIRE(d.count() == 1);
    REQUIRE(d.ledger()[0].code == AnomalyCode::ClippingRisk);
    REQUIRE(d.ledger()[0].severity == Severity::Error);
}

TEST_CASE("AnomalyDetector fires DeadLayer when sparsity > 0.9", "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.log_to_stderr = false});
    // Small max, small sigma, big sparsity -> only DeadLayer fires.
    d.evaluate(make_tensor(5, /*max=*/1.0f, /*sigma=*/1.0f, /*sparsity=*/0.95f));
    REQUIRE(d.count() == 1);
    REQUIRE(d.ledger()[0].code == AnomalyCode::DeadLayer);
}

TEST_CASE("AnomalyDetector fires LatencyHotspot when one sample > 3x median",
          "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.outlier_k = 6.0f,
                  .clipping_fp16_max = 65504.0f,
                  .clipping_fp32_max = 3.4e38f,
                  .dead_sparsity = 0.9f,
                  .hotspot_factor = 3.0f,
                  .log_to_stderr = false});

    // Establish a stable latency baseline of 100 us across 16 samples.
    for (int i = 0; i < 16; ++i) d.evaluate(make_latency(2, 100.0f));
    REQUIRE(d.count() == 0);

    // A 500 us sample should fire (~5x the rolling median of 100 us).
    d.evaluate(make_latency(2, 500.0f));
    REQUIRE(d.count() == 1);
    REQUIRE(d.ledger().back().code == AnomalyCode::LatencyHotspot);
}

TEST_CASE("AnomalyDetector is silent on LayerLatency until warmed up",
          "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.log_to_stderr = false});
    // First 7 samples shouldn't fire even if huge.
    for (int i = 0; i < 7; ++i) d.evaluate(make_latency(0, 9999.0f));
    REQUIRE(d.count() == 0);
}

TEST_CASE("AnomalyDetector clear() resets everything", "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.log_to_stderr = false});
    d.evaluate(make_tensor(0, /*max=*/60.0f, /*sigma=*/1.0f, /*sparsity=*/0.1f));
    d.evaluate(make_tensor(1, /*max=*/1.0f, /*sigma=*/1.0f, /*sparsity=*/0.95f));
    REQUIRE(d.count() == 2);
    d.clear();
    REQUIRE(d.count() == 0);

    // After clear, the latency window is empty too, so a 9999us sample
    // shouldn't fire because we haven't warmed up.
    d.evaluate(make_latency(0, 9999.0f));
    REQUIRE(d.count() == 0);
}

TEST_CASE("AnomalyDetector config can be queried", "[anomaly]") {
    AnomalyDetector d;
    auto c = d.config();
    REQUIRE(c.outlier_k == 50.0f);
    REQUIRE(c.dead_sparsity == 0.9f);
    REQUIRE(c.hotspot_factor == 3.0f);

    d.set_config({.outlier_k = 4.5f, .dead_sparsity = 0.5f, .hotspot_factor = 2.0f});
    REQUIRE(d.config().outlier_k == 4.5f);
    REQUIRE(d.config().dead_sparsity == 0.5f);
}

TEST_CASE("AnomalyDetector can fire multiple rules on the same packet",
          "[anomaly]") {
    AnomalyDetector d;
    d.set_config({.log_to_stderr = false});
    // Huge max that triggers BOTH OutlierFeature AND ClippingRisk at once.
    d.evaluate(make_tensor(0, /*max=*/70000.0f, /*sigma=*/1.0f, /*sparsity=*/0.5f));
    REQUIRE(d.count() == 2);
    // Newest entry is ClippingRisk (it is checked AFTER OutlierFeature in
    // evaluate()), but ledger order is insertion order.
    auto lg = d.ledger();
    bool saw_outlier = false, saw_clipping = false;
    for (auto& e : lg) {
        if (e.code == AnomalyCode::OutlierFeature) saw_outlier = true;
        if (e.code == AnomalyCode::ClippingRisk)   saw_clipping = true;
    }
    REQUIRE(saw_outlier);
    REQUIRE(saw_clipping);
}
