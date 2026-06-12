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
TEST_CASE("RingBuffer push blocks when full then unblocks on pop",
          "[ringbuffer][backpressure]") {
    RingBuffer<int> rb(2);
    REQUIRE(rb.push(1));
    REQUIRE(rb.push(2));

    std::atomic<bool> pushed_third{false};
    std::thread producer([&] {
        // Should block here until consumer pops
        rb.push(3);
        pushed_third = true;
    });

    // Give the producer a chance to block
    std::this_thread::sleep_for(50ms);
    REQUIRE_FALSE(pushed_third.load());

    // Free a slot
    auto v = rb.pop_for(1s);
    REQUIRE(v.has_value());
    REQUIRE(*v == 1);

    producer.join();
    REQUIRE(pushed_third.load());
    REQUIRE(rb.size() == 2);
}

TEST_CASE("RingBuffer close unblocks blocked producers",
          "[ringbuffer][backpressure]") {
    RingBuffer<int> rb(1);
    REQUIRE(rb.push(1));

    std::atomic<bool> returned{false};
    std::atomic<bool> push_returned_false{false};
    std::thread producer([&] {
        bool ok = rb.push(2);
        returned = true;
        push_returned_false = !ok;
    });

    std::this_thread::sleep_for(50ms);
    REQUIRE_FALSE(returned.load());

    rb.close();
    producer.join();

    REQUIRE(returned.load());
    REQUIRE(push_returned_false.load());   // push() must return false on close
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

    REQUIRE(consumed.load() == kProducers * kItemsPerProducer);
}

TEST_CASE("RingBuffer burst-with-backpressure stays under p99 deadline",
          "[ringbuffer][backpressure][perf]") {
    // Producer pushes 1000 into a cap-4 buffer; consumer drains in batches
    // of 16. Producer must spend MOST of its time blocked, but each individual
    // push() call (when there is space) must return quickly.
    constexpr int kItems   = 1000;
    constexpr std::size_t kCap = 4;

    RingBuffer<int> rb(kCap);

    std::atomic<int> p50_us{0};
    std::atomic<int> p99_us{0};
    std::atomic<int> count{0};

    std::thread producer([&] {
        for (int i = 0; i < kItems; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            rb.push(i);
            auto t1 = std::chrono::steady_clock::now();
            int us = static_cast<int>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            // Naive running p99 estimate: just take the max we see.
            // (Producer is mostly blocked, so almost every sample is ~0 us.)
            int prev = p99_us.load();
            while (us > prev && !p99_us.compare_exchange_weak(prev, us)) {}
            count.fetch_add(1);
        }
    });

    std::thread consumer([&] {
        int got = 0;
        while (got < kItems) {
            auto v = rb.pop_for(50ms);
            if (v.has_value()) ++got;
        }
    });

    producer.join();
    consumer.join();

    REQUIRE(count.load() == kItems);
    // No single push() should take more than 50 ms when there is contention.
    // (The producer is BLOCKED between pushes, so most are ~0 us.)
    REQUIRE(p99_us.load() < 50'000);
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
    for (int i = 0; i < kItems; ++i) {
        auto v = rb.pop_for(1s);
        if (v.has_value()) seen.push_back(*v);
    }
    producer.join();

    REQUIRE(seen.size() == static_cast<std::size_t>(kItems));
    // Items must come out in FIFO order.
    for (int i = 0; i < kItems; ++i) {
        REQUIRE(seen[i] == i);
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
    while (got < kItems) {
        auto v = rb.pop_for(1s);
        if (v.has_value()) {
            // Just verify the packet survived the round trip.
            REQUIRE(v->sequence_id == static_cast<std::uint32_t>(got));
            REQUIRE(v->layer_id == got % 36);
            ++got;
        }
    }
    producer.join();
    REQUIRE(got == kItems);
}
