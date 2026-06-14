// =============================================================================
//  RingBuffer.hpp
//  -----------------------------------------------------------------------------
//  Thread-safe bounded FIFO used as the producer/consumer queue between
//  the llama.cpp interceptor (producer) and the TUI (consumer).
//
//  Implementation: std::deque + std::mutex + std::condition_variable.
//  When full, push() BLOCKS the producer (back-pressure — the model slows
//  down rather than blowing the heap).
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

    // Non-blocking push: if buffer is at max capacity, drop the oldest item. Returns false if shutdown.
    bool push(const T& item) {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) return false;
        if (buf_.size() >= max_size()) {
            buf_.pop_front(); // Silently drop oldest data instead of blocking
        }
        buf_.push_back(item);
        cv_not_empty_.notify_one();
        return true;
    }

    // Non-blocking push: drops the item if the buffer is at 90% capacity or full.
    bool try_push(const T& item) {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_ || buf_.size() >= max_size()) return false;
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

    // Non-blocking pop: returns the front item, or nullopt if empty or closed.
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_ || buf_.empty()) return std::nullopt;
        T item = std::move(buf_.front());
        buf_.pop_front();
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
    std::size_t max_size() const noexcept {
        if (capacity_ < 10) {
            return capacity_;
        }
        return (capacity_ * 9) / 10;
    }

    mutable std::mutex      mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::deque<T>           buf_;
    std::size_t             capacity_;
    bool                    closed_ = false;
};

} // namespace llm_tui
