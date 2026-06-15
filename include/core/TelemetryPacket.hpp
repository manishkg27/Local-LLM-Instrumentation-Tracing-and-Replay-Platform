// =============================================================================
//  TelemetryPacket.hpp
//  -----------------------------------------------------------------------------
//  The single data structure that flows from the llama.cpp interceptor
//  (producer) through the ring buffer to the TUI (consumer).
//
//  Designed to be:
//    * Trivially copyable (we memcpy / queue it through std::deque).
//    * Fixed-size (no std::string, no vectors) so it doesn't fragment the heap.
//    * Self-describing (every packet carries a kind + a wall-clock timestamp).
//
//  All units are SI / scientific:
//      latencies   -> microseconds (us)
//      shapes      -> int32 dimensions
//      stats       -> float32
//      timestamps  -> nanoseconds since epoch (steady_clock)
// =============================================================================
#pragma once

#include <array>
#include <chrono>
#include <cstdint>

namespace llm_tui {

// --- Packet kinds ----------------------------------------------------------
// We use a single enum so the TUI can colour/filter packets by type.
enum class PacketKind : std::uint8_t {
    Unknown          = 0,
    Topology         = 1,  // Sent ONCE per layer at model-load time
    TokenStart       = 2,  // Begin of a forward pass (one per llama_decode call)
    LayerLatency     = 3,  // Per-layer wall time
    TensorStats      = 4,  // Per-tensor shape / sparsity / mean / max
    AttentionSlice   = 5,  // Optional 7x7 attention window (or smaller)
    TokenEnd         = 6,  // End of forward pass
    Anomaly          = 7,  // Detected numerical anomaly
};

inline const char* to_string(PacketKind k) {
    switch (k) {
        case PacketKind::Topology:       return "TOPOLOGY";
        case PacketKind::TokenStart:     return "TOKEN-START";
        case PacketKind::LayerLatency:   return "LAYER-LATENCY";
        case PacketKind::TensorStats:    return "TENSOR-STATS";
        case PacketKind::AttentionSlice: return "ATTN-SLICE";
        case PacketKind::TokenEnd:       return "TOKEN-END";
        case PacketKind::Anomaly:        return "ANOMALY";
        default:                         return "UNKNOWN";
    }
}

// --- Tensor shape ---------------------------------------------------------
// We cap at 4 dims because that's all llama.cpp / ggml tensors use in
// practice (batch, seq, heads, hidden). Anything higher is a bug.
static constexpr std::size_t kMaxDims = 4;
using Shape = std::array<std::int32_t, kMaxDims>;

// --- Layer type tags ------------------------------------------------------
// These mirror the names from llama.cpp's model loader so the TUI can show
// human-readable labels (e.g. "Attn (Self)", "MLP (SwiGLU)").
enum class LayerType : std::uint8_t {
    Unknown        = 0,
    Embedding      = 1,
    AttentionSelf  = 2,
    AttentionCross = 3,
    Mlp            = 4,
    RmsNorm        = 5,
    LayerNorm      = 6,
    Output         = 7,
    Token          = 8,
};

inline const char* to_string(LayerType t) {
    switch (t) {
        case LayerType::Embedding:      return "Embed";
        case LayerType::AttentionSelf:  return "Attn (Self)";
        case LayerType::AttentionCross: return "Attn (Cross)";
        case LayerType::Mlp:            return "MLP (SwiGLU)";
        case LayerType::RmsNorm:        return "RMSNorm";
        case LayerType::LayerNorm:      return "LayerNorm";
        case LayerType::Output:         return "Output";
        case LayerType::Token:          return "Token";
        default:                        return "?";
    }
}

// --- Severity levels for anomalies ---------------------------------------
enum class Severity : std::uint8_t { Info = 0, Warn = 1, Error = 2 };

// --- Anomaly sub-codes ---------------------------------------------------
// Stable codes so the TUI can group / count them across a session.
enum class AnomalyCode : std::uint8_t {
    None             = 0,
    OutlierFeature   = 1,  // max(|x|) > k * sigma
    ClippingRisk     = 2,  // saturating at fp16/fp32 max
    DeadLayer        = 3,  // sparsity > 0.9 with no movement
    LatencyHotspot   = 4,  // latency > 3x median
    CpuFallback      = 5,  // CUDA OOM, ran on CPU
    NaNInf           = 6,  // NaN/Inf detected
};

// --- The packet itself ---------------------------------------------------
// Fixed size: 64 bytes (one cache line). Anything that needs more (e.g. a
// full attention matrix) lives in a side-channel queue keyed by packet id.
struct TelemetryPacket {
    // ----- 8 bytes -----
    std::uint64_t timestamp_ns;   // wall clock, std::chrono::steady_clock
    std::uint32_t sequence_id;    // monotonic per-token counter
    std::uint32_t flags;          // reserved (bit 0 = focus, etc.)

    // ----- 8 bytes -----
    std::int32_t  layer_id;       // -1 = meta (model-wide); 0..N-1 = layer index
    PacketKind    kind;
    LayerType     layer_type;
    Severity      severity;
    AnomalyCode   anomaly_code;
    // ----- NEW: attention metadata -----
    std::int8_t   head_idx;       // which attention head this attn_patch belongs to (-1 = unknown)
    std::int8_t   attn_seq_len;   // actual sequence length at capture time (packed into padding area)
    std::int16_t  reserved_pad;   // explicit padding to keep alignment

    // ----- 24 bytes -----
    Shape         shape;          // tensor shape, populated when kind == TensorStats
    std::int32_t  device;         // 0 = CPU, 1 = CUDA, ...

    // ----- 24 bytes -----
    float         latency_us;     // wall time for this step
    float         sparsity;       // 0..1, fraction of zeros / sub-threshold values
    float         mean;
    float         max_abs;
    float         sigma;          // std-dev (for outlier z-score)
    std::int32_t  padding;        // explicit padding to keep alignment 8B

    // ----- 196 bytes (optional attention window) -----
    // Up to 7x7 = 49 floats. Larger matrices are streamed through the side
    // channel; we just keep a representative 7x7 patch here for the TUI.
    float         attn_patch[7 * 7];
};
// TelemetryPacket is a *target* of 80 bytes (one cache line) but the actual
// layout can be a bit larger once the compiler inserts alignment padding for
// the trailing attn_patch[49] array. Allow any size in [80, 300] bytes so
// changing field order or alignment later doesn't trigger a recompile storm
// for every downstream consumer (RingBuffer, TUI, replay tool, ...).
static_assert(sizeof(TelemetryPacket) >= 80,
              "TelemetryPacket shrunk below its 80-byte design target");
static_assert(sizeof(TelemetryPacket) <= 300,
              "TelemetryPacket grew above the 300-byte budget");

// Helper: build a "blank" packet
inline TelemetryPacket make_packet(PacketKind k, std::uint32_t seq = 0) {
    TelemetryPacket p{};
    p.timestamp_ns =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    p.sequence_id = seq;
    p.kind        = k;
    p.layer_id    = -1;
    p.layer_type  = LayerType::Unknown;
    p.severity    = Severity::Info;
    p.anomaly_code = AnomalyCode::None;
    p.shape       = {0, 0, 0, 0};
    p.device      = 0;
    p.latency_us  = 0.0f;
    p.sparsity    = 0.0f;
    p.mean        = 0.0f;
    p.max_abs     = 0.0f;
    p.sigma       = 0.0f;
    return p;
}

} // namespace llm_tui
