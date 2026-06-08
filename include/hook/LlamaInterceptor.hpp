// =============================================================================
//  LlamaInterceptor.hpp
//  -----------------------------------------------------------------------------
//  Thin RAII wrapper around llama.cpp's public C API.
//
//  Day 1: load a GGUF model, decode 1 token, print layer topology.
//  Day 3: wrap llama_decode with chrono timing, build ModelTopology,
//         push TelemetryPackets to the RingBuffer.
//
//  We do NOT modify llama.cpp source. We just call its public C API and
//  add instrumentation around the calls. This satisfies the "non-invasive"
//  requirement of the project.
// =============================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/TelemetryPacket.hpp"

struct llama_model;
struct llama_context;
struct llama_sampler;

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
    LlamaInterceptor() = default;
    ~LlamaInterceptor();

    LlamaInterceptor(const LlamaInterceptor&)            = delete;
    LlamaInterceptor& operator=(const LlamaInterceptor&) = delete;

    // Load a GGUF file from disk. Returns true on success. On failure,
    // an error message is logged via spdlog and `loaded()` returns false.
    bool load(const std::string& gguf_path,
              int n_ctx = 2048,
              int n_threads = 0 /* 0 = auto */);

    // Day 1: decode a single token, print the result, return latency in us.
    // Day 3: full implementation that pushes TelemetryPackets.
    float decode_one(const std::string& prompt);

    const ModelTopology& topology() const { return topology_; }
    bool loaded() const { return model_ != nullptr && ctx_ != nullptr; }

private:
    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    llama_sampler* smpl_  = nullptr;
    ModelTopology  topology_;
};

} // namespace llm_tui
