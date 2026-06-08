// =============================================================================
//  RingBuffer.hpp
//  -----------------------------------------------------------------------------
//  Thread-safe bounded FIFO used as the producer/consumer queue between
//  the llama.cpp interceptor (producer) and the TUI (consumer).
//
//  Implementation: std::deque + std::mutex + std::condition_variable.
//  When full, push() BLOCKS the producer (back-pressure — the model slows
//  down rather than blowing the heap).
//
//  Day 1: this is a STUB (interface only). Full implementation + tests
//  land on Day 2.
// =============================================================================
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

#include "core/TelemetryPacket.hpp"

namespace llm_tui {

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity) : capacity_(capacity) {}

    RingBuffer(const RingBuffer&)            = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // Block until the buffer has space, then push. Returns false if shutdown.
    bool push(const T& item) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk, [&] { return closed_ || buf_.size() < capacity_; });
        if (closed_) return false;
        buf_.push_back(item);
        lk.unlock();
        cv_not_empty_.notify_one();
        return true;
    }

    // Non-blocking push: drops the item if the buffer is full. Useful for
    // stats packets where we don't want to stall the model.
    bool try_push(const T& item) {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_ || buf_.size() >= capacity_) return false;
        buf_.push_back(item);
        cv_not_empty_.notify_one();
        return true;
    }

    // Block up to `timeout` for an item. Returns nullopt on timeout.
    template <typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!cv_not_empty_.wait_for(lk, timeout, [&] { return closed_ || !buf_.empty(); })) {
            return std::nullopt;
        }
        if (buf_.empty()) return std::nullopt;
        T item = std::move(buf_.front());
        buf_.pop_front();
        lk.unlock();
        cv_not_full_.notify_one();
        return item;
    }

    // Snapshot all items (non-blocking). Used by the TUI to render.
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        return {buf_.begin(), buf_.end()};
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return buf_.size();
    }

    std::size_t capacity() const noexcept { return capacity_; }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::deque<T>           buf_;
    std::size_t             capacity_;
    bool                    closed_ = false;
};

} // namespace llm_tui
