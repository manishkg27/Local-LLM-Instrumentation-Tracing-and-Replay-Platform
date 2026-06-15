// =============================================================================
//  LlamaInterceptor.hpp
//  -----------------------------------------------------------------------------
//  Thin RAII wrapper around llama.cpp's public C API.
//
//  We do NOT modify llama.cpp source. We just call its public C API and
//  add instrumentation around the calls.
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "core/AnomalyDetector.hpp"
#include "core/AttentionSnapshot.hpp"
#include "core/RingBuffer.hpp"
#include "core/TelemetryPacket.hpp"

struct llama_model;
struct llama_context;
struct llama_sampler;
struct ggml_tensor;

namespace llm_tui {

// --- Model topology -----------------------------------------------------
// A small POD struct summarising the model's structure. The TUI's Panel 1
// (Model Topology) renders this as a collapsible tree.
struct ModelTopology {
    std::string name;             // e.g. "Qwen2.5-Coder-3B-Instruct"
    std::int32_t n_layer    = 0;  // transformer blocks
    std::int32_t n_head     = 0;  // query heads
    std::int32_t n_head_kv  = 0;  // key/value heads (GQA)
    std::int32_t n_embd     = 0;  // hidden size
    std::int32_t n_vocab    = 0;  // vocab size
    std::uint64_t n_params  = 0;  // total parameters
    std::int32_t device     = 0;  // 0 = CPU, 1 = CUDA
    std::string  device_name;
};

class LlamaInterceptor {
public:
    LlamaInterceptor();
    ~LlamaInterceptor();

    LlamaInterceptor(const LlamaInterceptor&)            = delete;
    LlamaInterceptor& operator=(const LlamaInterceptor&) = delete;

    // Load a GGUF file from disk. Returns true on success. On failure,
    // an error message is logged via spdlog and `loaded()` returns false.
    bool load(const std::string& gguf_path,
              int n_ctx = 2048,
              int n_threads = 0 /* 0 = auto */);

    // Day 1: decode a single token, print the result, return latency in us.
    // Day 2: also pushes a TensorStart / LayerLatency / TokenEnd packet to
    //        the optional sink and runs them through the AnomalyDetector.
    // Day 3: per-layer timing; multiple packets per call.
    // Day 7: Changed to generate up to max_tokens and pass text to a callback.
    void generate(const std::string& prompt, std::function<void(const std::string&)> on_token = nullptr);

    // Wire a sink ring-buffer. Optional — without it, packets just go to
    // the AnomalyDetector and are dropped after evaluation. Multiple calls
    // replace the previous sink.
    void set_sink(RingBuffer<TelemetryPacket>* sink) { sink_ = sink; }

    // Wire an atomic pointer from the TUI to dynamically read the selected head.
    void set_active_head_ptr(std::atomic<int>* ptr) { active_head_ptr_ = ptr; }

    // Read-only accessors used by the TUI panels.
    const ModelTopology& topology() const { return topology_; }
    AnomalyDetector&     detector()       { return detector_; }
    const AnomalyDetector& detector() const { return detector_; }
    bool loaded() const { return model_ != nullptr && ctx_ != nullptr; }

    // Access the full attention matrix cache (thread-safe via shared_mutex).
    AttentionCache& attn_cache() { return attn_cache_; }
    const AttentionCache& attn_cache() const { return attn_cache_; }

private:
    // Helper: build a Topology packet from this->topology_ and dispatch it
    // through both the sink and the detector. Called once at the end of
    // load().
    void emit_topology_packet();
    bool on_eval(struct ggml_tensor* t, bool ask);
    void metrics_worker_loop();

    struct PendingMetrics {
        TelemetryPacket p;
        std::vector<float> data_sample;
    };

    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    llama_sampler* smpl_  = nullptr;
    ModelTopology  topology_;
    AnomalyDetector detector_;
    RingBuffer<TelemetryPacket>* sink_ = nullptr;
    std::uint32_t  seq_counter_ = 0;
    std::chrono::steady_clock::time_point last_tensor_time_;
    std::vector<float> layer_latencies_us_;
    AttentionCache attn_cache_;   // full attention matrix side-channel
    std::atomic<int>* active_head_ptr_ = nullptr;

    // Async metrics computation
    RingBuffer<PendingMetrics> metrics_queue_{1024};
    std::thread metrics_worker_;
    std::atomic<bool> worker_running_{true};
};

} // namespace llm_tui
