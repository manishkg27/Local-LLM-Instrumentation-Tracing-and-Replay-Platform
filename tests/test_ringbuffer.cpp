// =============================================================================
//  test_ringbuffer.cpp
//  -----------------------------------------------------------------------------
//  Catch2 v3 unit tests for RingBuffer<TelemetryPacket>.
//  Runs via `ctest` (see CMakeLists.txt).
// =============================================================================
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "core/RingBuffer.hpp"
#include "core/TelemetryPacket.hpp"

using namespace llm_tui;
using namespace std::chrono_literals;

TEST_CASE("RingBuffer basic push/pop", "[ringbuffer]") {
    RingBuffer<int> rb(4);
    REQUIRE(rb.capacity() == 4);
    REQUIRE(rb.size() == 0);

    REQUIRE(rb.push(1));
    REQUIRE(rb.push(2));
    REQUIRE(rb.push(3));
    REQUIRE(rb.size() == 3);

    auto v = rb.snapshot();
    REQUIRE(v == std::vector<int>{1, 2, 3});
}

TEST_CASE("RingBuffer try_push drops on overflow", "[ringbuffer]") {
    RingBuffer<int> rb(2);
    REQUIRE(rb.try_push(1));
    REQUIRE(rb.try_push(2));
    REQUIRE_FALSE(rb.try_push(3));   // full -> drop
    REQUIRE(rb.size() == 2);
}

TEST_CASE("RingBuffer pop_for returns on data", "[ringbuffer]") {
    RingBuffer<int> rb(2);
    rb.push(42);
    auto v = rb.pop_for(100ms);
    REQUIRE(v.has_value());
    REQUIRE(*v == 42);
    REQUIRE(rb.size() == 0);
}

TEST_CASE("RingBuffer pop_for times out when empty", "[ringbuffer]") {
    RingBuffer<int> rb(2);
    auto v = rb.pop_for(50ms);
    REQUIRE_FALSE(v.has_value());
}

TEST_CASE("RingBuffer close unblocks consumers", "[ringbuffer]") {
    RingBuffer<int> rb(1);
    std::atomic<int> got{-1};
    std::thread t([&] {
        auto v = rb.pop_for(10s);
        got = v.value_or(-1);
    });
    std::this_thread::sleep_for(20ms);
    rb.close();
    t.join();
    REQUIRE(got == -1);
}

TEST_CASE("RingBuffer with TelemetryPacket is trivially copyable",
          "[ringbuffer][packet]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<TelemetryPacket>);
    RingBuffer<TelemetryPacket> rb(8);
    auto p = make_packet(PacketKind::TensorStats, /*seq=*/1);
    p.layer_id = 7;
    p.shape = {1, 32, 4096, 0};
    p.max_abs = 3.14f;
    REQUIRE(rb.push(p));
    auto out = rb.pop_for(100ms);
    REQUIRE(out.has_value());
    REQUIRE(out->layer_id == 7);
    REQUIRE(out->shape[1] == 32);
    REQUIRE(out->max_abs == 3.14f);
}
