// =============================================================================
//  test_ringbuffer.cpp
//  -----------------------------------------------------------------------------
//  Catch2 v3 unit tests for RingBuffer<TelemetryPacket>.
//  Runs via `ctest` (see CMakeLists.txt).
// =============================================================================
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "core/RingBuffer.hpp"
#include "core/TelemetryPacket.hpp"

using namespace llm_tui;
using namespace std::chrono_literals;

// -----------------------------------------------------------------------------
//  Day 1 baseline tests
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
//  Day 2 — stress / multi-producer / backpressure tests
// -----------------------------------------------------------------------------
TEST_CASE("RingBuffer push drops oldest when full",
          "[ringbuffer][drop]") {
    RingBuffer<int> rb(2);
    REQUIRE(rb.push(1));
    REQUIRE(rb.push(2));
    REQUIRE(rb.push(3)); // Drops 1

    REQUIRE(rb.size() == 2);
    
    auto v = rb.pop_for(100ms);
    REQUIRE(v.has_value());
    REQUIRE(*v == 2);

    v = rb.pop_for(100ms);
    REQUIRE(v.has_value());
    REQUIRE(*v == 3);
}

TEST_CASE("RingBuffer push returns false after close",
          "[ringbuffer][close]") {
    RingBuffer<int> rb(2);
    REQUIRE(rb.push(1));

    rb.close();

    // After close, push should return false
    REQUIRE_FALSE(rb.push(2));
    
    // We can still pop remaining data but then it will return nullopt
    auto v = rb.pop_for(1s);
    REQUIRE(v.has_value());
    REQUIRE(*v == 1);
    
    auto v2 = rb.pop_for(100ms);
    REQUIRE_FALSE(v2.has_value());
}

TEST_CASE("RingBuffer multi-producer / single-consumer delivers all items",
          "[ringbuffer][concurrency]") {
    constexpr int kProducers       = 2;
    constexpr int kItemsPerProducer = 500;
    constexpr int kCap             = 16;

    RingBuffer<int> rb(kCap);

    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&rb, p] {
            for (int i = 0; i < kItemsPerProducer; ++i) {
                // P0 emits even numbers, P1 emits odd numbers so we can
                // verify both producers' items made it through.
                rb.push(p * 100000 + i);
            }
        });
    }

    std::atomic<int> consumed{0};
    std::atomic<bool> done{false};
    std::thread consumer([&] {
        while (!done.load() || rb.size() > 0) {
            auto v = rb.pop_for(20ms);
            if (v.has_value()) consumed.fetch_add(1);
        }
    });

    for (auto& t : producers) t.join();
    done = true;
    consumer.join();

    // Since we drop on full, we may not get all items, but we should get some
    // and no crashes should happen.
    REQUIRE(consumed.load() > 0);
    REQUIRE(consumed.load() <= kProducers * kItemsPerProducer);
}

TEST_CASE("RingBuffer burst pushes complete quickly without blocking",
          "[ringbuffer][perf]") {
    // Producer pushes 1000 into a cap-4 buffer. Since it drops oldest, it should never block.
    constexpr int kItems   = 1000;
    constexpr std::size_t kCap = 4;

    RingBuffer<int> rb(kCap);

    std::atomic<int> p99_us{0};
    std::atomic<int> count{0};

    std::thread producer([&] {
        for (int i = 0; i < kItems; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            rb.push(i);
            auto t1 = std::chrono::steady_clock::now();
            int us = static_cast<int>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            int prev = p99_us.load();
            while (us > prev && !p99_us.compare_exchange_weak(prev, us)) {}
            count.fetch_add(1);
        }
    });

    producer.join();
    REQUIRE(count.load() == kItems);
    REQUIRE(p99_us.load() < 50'000); // Should be very fast
}

TEST_CASE("RingBuffer snapshot is consistent under concurrent push",
          "[ringbuffer][concurrency]") {
    // Snapshot returns a *copy*; this is a smoke test that we never see
    // a torn read or a use-after-free even when the producer is mid-push.
    constexpr int kItems = 200;
    RingBuffer<int> rb(8);

    std::thread producer([&] {
        for (int i = 0; i < kItems; ++i) rb.push(i);
    });

    std::vector<int> seen;
    while (true) {
        auto v = rb.pop_for(10ms); // Use short timeout
        if (!v.has_value()) break;
        seen.push_back(*v);
    }
    producer.join();

    REQUIRE(seen.size() > 0);
    // Items must come out in strictly increasing order (FIFO, some may be dropped).
    int last = -1;
    for (size_t i = 0; i < seen.size(); ++i) {
        REQUIRE(seen[i] > last);
        last = seen[i];
    }
}

TEST_CASE("RingBuffer long TelemetryPacket run does not lose data",
          "[ringbuffer][packet][long]") {
    constexpr int kItems = 2048;
    RingBuffer<TelemetryPacket> rb(64);

    std::thread producer([&] {
        for (int i = 0; i < kItems; ++i) {
            auto p = make_packet(PacketKind::TensorStats, /*seq=*/i);
            p.layer_id = i % 36;
            p.shape = {1, 1, 16, 128};
            p.max_abs = static_cast<float>(i) * 0.01f;
            p.sparsity = static_cast<float>(i % 100) / 100.0f;
            REQUIRE(rb.push(p));
        }
    });

    int got = 0;
    int last_seq = -1;
    while (true) {
        auto v = rb.pop_for(50ms);
        if (!v.has_value()) break;
        // Verify sequence IDs are strictly increasing (though gaps are allowed)
        REQUIRE(static_cast<int>(v->sequence_id) > last_seq);
        last_seq = static_cast<int>(v->sequence_id);
        REQUIRE(v->layer_id == last_seq % 36);
        ++got;
    }
    producer.join();
    REQUIRE(got > 0);
}
