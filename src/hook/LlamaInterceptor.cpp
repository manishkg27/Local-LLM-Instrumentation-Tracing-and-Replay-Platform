// =============================================================================
//  LlamaInterceptor.cpp
// =============================================================================
#include "hook/LlamaInterceptor.hpp"
#include "llama-model.h"

#include <llama.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <chrono>
#include <cstring>
#include <thread>
#include <algorithm>

namespace llm_tui {

namespace {
std::shared_ptr<spdlog::logger> interceptor_log() {
    static auto lg = spdlog::stdout_color_mt("llama");
    return lg;
}

// Push a packet to BOTH the optional sink AND the in-process AnomalyDetector.
// This is the single funnel for Day 2; on D6 the consumer thread takes over.
void dispatch(const TelemetryPacket& pkt,
              RingBuffer<TelemetryPacket>* sink,
              AnomalyDetector& detector) {
    detector.evaluate(pkt);
    if (sink) sink->push(pkt);  // blocks at cap (back-pressure on D6)
}

int get_layer_id(const char* name) {
    if (name == nullptr || *name == '\0') return -1;

    // Find the last occurrence of '-'
    const char* last_hyphen = std::strrchr(name, '-');
    if (last_hyphen != nullptr) {
        char* end = nullptr;
        long val = std::strtol(last_hyphen + 1, &end, 10);
        if (end != last_hyphen + 1 && *end == '\0') {
            return static_cast<int>(val);
        }
    }

    // Fallback: check if the weight tensor style is used (e.g., "blk.0.attn_q")
    if (std::strncmp(name, "blk.", 4) == 0) {
        char* end = nullptr;
        long val = std::strtol(name + 4, &end, 10);
        if (end != name + 4 && *end == '.') {
            return static_cast<int>(val);
        }
    }
    return -1;
}

std::string get_tensor_base_name(const char* name) {
    if (name == nullptr || *name == '\0') return "";
    const char* last_hyphen = std::strrchr(name, '-');
    if (last_hyphen != nullptr) {
        return std::string(name, last_hyphen - name);
    }
    return name;
}

LayerType deduce_layer_type(const llama_layer& layer) {
    if (layer.wq) {
        return LayerType::AttentionSelf;
    }
    if (layer.wq_cross) {
        return LayerType::AttentionCross;
    }
    if (layer.ffn_gate) {
        return LayerType::Mlp;
    }
    if (layer.attn_norm) {
        return LayerType::RmsNorm;
    }
    return LayerType::Unknown;
}
} // namespace

LlamaInterceptor::LlamaInterceptor() = default;

LlamaInterceptor::~LlamaInterceptor() {
    if (smpl_)  llama_sampler_free(smpl_);
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

bool LlamaInterceptor::load(const std::string& gguf_path, int n_ctx, int n_threads) {
    auto log = interceptor_log();
    log->info("LlamaInterceptor::load(\"{}\", n_ctx={}, threads={})",
              gguf_path, n_ctx, n_threads);

    // ---- Backend init ----
    llama_backend_init();

    // ---- Model params ----
    auto mparams = llama_model_default_params();
    // We keep defaults; persona model is small enough.
    mparams.n_gpu_layers = 0;  // CPU-only for Day 1 (no CUDA detected)

    model_ = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model_) {
        log->error("llama_model_load_from_file failed for \"{}\"", gguf_path);
        return false;
    }

    // ---- Context params ----
    auto cparams = llama_context_default_params();
    cparams.n_ctx        = n_ctx;
    cparams.n_threads    = (n_threads > 0) ? n_threads
                                          : static_cast<int>(std::thread::hardware_concurrency());
    cparams.n_threads_batch = cparams.n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval = [](struct ggml_tensor* t, bool ask, void* user_data) -> bool {
        auto* self = static_cast<LlamaInterceptor*>(user_data);
        return self->on_eval(t, ask);
    };
    cparams.cb_eval_user_data = this;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        log->error("llama_init_from_model failed");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    // ---- Sampler (greedy for now) ----
    auto sparams = llama_sampler_chain_default_params();
    smpl_ = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());

    // ---- Populate topology ----
    const auto* vocab = llama_model_get_vocab(model_);
    char buf[256] = {0}; int n_desc = llama_model_desc(model_, buf, sizeof(buf));
    if (n_desc > 0) {
        topology_.name = buf;
    } else {
        topology_.name = "unknown";
    }
    topology_.n_layer   = llama_model_n_layer(model_);
    topology_.n_head    = llama_model_n_head(model_);
    topology_.n_head_kv = llama_model_n_head_kv(model_);
    topology_.n_embd    = llama_model_n_embd(model_);
    topology_.n_vocab   = llama_vocab_n_tokens(vocab);
    topology_.n_params  = llama_model_n_params(model_);
    topology_.device    = 0;  // CPU
    topology_.device_name = "CPU (AVX2)";

    layer_latencies_us_.resize(topology_.n_layer, 0.0f);

    log->info("Loaded model \"{}\"", topology_.name);
    log->info("  n_layer  = {}", topology_.n_layer);
    log->info("  n_head   = {}  (kv heads: {})", topology_.n_head, topology_.n_head_kv);
    log->info("  n_embd   = {}", topology_.n_embd);
    log->info("  n_vocab  = {}", topology_.n_vocab);
    log->info("  n_params = {:.2f} B", topology_.n_params / 1e9);
    log->info("  device   = {}", topology_.device_name);

    // Day 2: emit a single topology packet so the rest of the pipeline
    // (sink, detector) has something to chew on even before D3 stats land.
    emit_topology_packet();
    return true;
}

void LlamaInterceptor::emit_topology_packet() {
    TelemetryPacket p = make_packet(PacketKind::Topology, /*seq=*/++seq_counter_);
    p.layer_id   = -1;
    p.layer_type = LayerType::Unknown;
    p.shape      = {topology_.n_layer, topology_.n_head, topology_.n_embd, topology_.n_vocab};
    p.device     = topology_.device;
    // Stash the parameter count in the sigma slot as a cheap visible signal.
    p.sigma      = static_cast<float>(static_cast<double>(topology_.n_params) / 1e9);
    dispatch(p, sink_, detector_);

    if (model_ && !model_->layers.empty()) {
        int num_layers = std::min(static_cast<int>(model_->layers.size()), topology_.n_layer);
        for (int i = 0; i < num_layers; ++i) {
            const auto& layer = model_->layers[i];
            TelemetryPacket lp = make_packet(PacketKind::Topology, /*seq=*/++seq_counter_);
            lp.layer_id   = i;
            lp.layer_type = deduce_layer_type(layer);
            int32_t layer_ff = model_->hparams.n_ff(i);
            lp.shape      = {topology_.n_head, topology_.n_head_kv, topology_.n_embd, layer_ff};
            lp.device     = topology_.device;
            dispatch(lp, sink_, detector_);
        }
    }
}

bool LlamaInterceptor::on_eval(struct ggml_tensor* t, bool ask) {
    if (ask) {
        return true;
    }
    auto now = std::chrono::steady_clock::now();
    float elapsed_us = std::chrono::duration<float, std::micro>(now - last_tensor_time_).count();
    last_tensor_time_ = now;

    const char* name = t->name ? t->name : "";
    int layer_id = get_layer_id(name);

    if (layer_id >= 0 && layer_id < static_cast<int>(layer_latencies_us_.size())) {
        layer_latencies_us_[layer_id] += elapsed_us;
    }

    // Capture Tensor statistics for designated intermediate nodes
    if (t->data != nullptr && t->type == GGML_TYPE_F32) {
        std::string base_name = get_tensor_base_name(name);
        if (base_name == "Qcur" || base_name == "kqv_out" || base_name == "ffn_down" || base_name == "norm" || base_name == "kq_soft_max") {
            int64_t n = ggml_nelements(t);
            if (n > 0) {
                const float* data = (const float*)t->data;
                double sum = 0.0;
                double sum_sq = 0.0;
                float max_abs = 0.0f;
                int64_t zero_count = 0;
                
                // Limit sampling size to 4096 elements to keep timing overhead negligible on CPU
                int64_t step = 1;
                int64_t limit = n;
                if (n > 4096) {
                    step = n / 4096;
                    limit = 4096 * step;
                }
                
                int64_t counted = 0;
                for (int64_t i = 0; i < limit; i += step) {
                    float val = data[i];
                    sum += val;
                    sum_sq += val * val;
                    float abs_val = std::abs(val);
                    if (abs_val > max_abs) {
                        max_abs = abs_val;
                    }
                    if (abs_val < 1e-6f) {
                        zero_count++;
                    }
                    counted++;
                }
                
                float mean = static_cast<float>(sum / counted);
                float variance = static_cast<float>((sum_sq / counted) - (mean * mean));
                float sigma = std::sqrt(std::max(0.0f, variance));
                float sparsity = static_cast<float>(zero_count) / counted;
                
                TelemetryPacket p = make_packet(PacketKind::TensorStats, ++seq_counter_);
                p.layer_id = layer_id;
                
                if (base_name == "Qcur" || base_name == "kqv_out" || base_name == "kq_soft_max") {
                    p.layer_type = LayerType::AttentionSelf;
                } else if (base_name == "ffn_down") {
                    p.layer_type = LayerType::Mlp;
                } else if (base_name == "norm") {
                    p.layer_type = LayerType::RmsNorm;
                }
                
                p.shape = {
                    static_cast<int32_t>(t->ne[0]),
                    static_cast<int32_t>(t->ne[1]),
                    static_cast<int32_t>(t->ne[2]),
                    static_cast<int32_t>(t->ne[3])
                };
                
                p.sparsity = sparsity;
                p.mean = mean;
                p.max_abs = max_abs;
                p.sigma = sigma;
                p.device = 0;
                
                // Extract 4x4 attention slice from kq_soft_max activations if available
                if (base_name == "kq_soft_max" && t->ne[0] > 0 && t->ne[1] > 0) {
                    int64_t n_kv = t->ne[0];
                    int64_t n_q = t->ne[1];
                    int r_max = std::min(static_cast<int64_t>(7), n_q);
                    int c_max = std::min(static_cast<int64_t>(7), n_kv);
                    
                    int r_start = std::max(static_cast<int64_t>(0), n_q - 7);
                    int c_start = std::max(static_cast<int64_t>(0), n_kv - 7);
                    
                    float patch_max = 0.0f;
                    for (int r = 0; r < r_max; ++r) {
                        for (int c = 0; c < c_max; ++c) {
                            patch_max = std::max(patch_max, std::abs(data[(r_start + r) * n_kv + (c_start + c)]));
                        }
                    }
                    if (patch_max < 1e-12f) {
                        patch_max = 1.0f;
                    }
                    
                    // zero out first
                    std::fill(std::begin(p.attn_patch), std::end(p.attn_patch), 0.0f);
                    
                    // Anchor to the bottom-right of the 7x7 patch buffer
                    int patch_r_start = 7 - r_max;
                    int patch_c_start = 7 - c_max;
                    
                    for (int r = 0; r < r_max; ++r) {
                        for (int c = 0; c < c_max; ++c) {
                            p.attn_patch[(patch_r_start + r) * 7 + (patch_c_start + c)] = 
                                std::abs(data[(r_start + r) * n_kv + (c_start + c)]) / patch_max;
                        }
                    }
                }
                dispatch(p, sink_, detector_);
                
                // Artificially slow down execution so the TUI can render the active node progression
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    return true;
}

void LlamaInterceptor::generate(const std::string& prompt, std::function<void(const std::string&)> on_token) {
    auto log = interceptor_log();
    if (!loaded()) {
        log->error("generate called before successful load()");
        return;
    }

    // Clear KV cache so we can start decoding at position 0
    llama_memory_seq_rm(llama_get_memory(ctx_), -1, -1, -1);

    // ---- Tokenize the prompt ----
    const auto* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), tokens.data(), static_cast<int>(tokens.size()), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), tokens.data(), static_cast<int>(tokens.size()), true, true);
    }
    tokens.resize(n_tokens);
    log->info("Prompt \"{}\" tokenized to {} tokens", prompt, n_tokens);

    int max_tokens = 64; // Max tokens to generate
    int pos = 0;

    for (int step = 0; step < max_tokens; ++step) {
        // ---- TokenStart packet (one per decode call) ----
        {
            TelemetryPacket p = make_packet(PacketKind::TokenStart, ++seq_counter_);
            p.layer_id   = -1;
            p.layer_type = LayerType::Token;
            p.shape      = {1, 0, 0, 0};
            dispatch(p, sink_, detector_);
        }

        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        batch.n_tokens = tokens.size();
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == tokens.size() - 1);
        }

        std::fill(layer_latencies_us_.begin(), layer_latencies_us_.end(), 0.0f);
        last_tensor_time_ = std::chrono::steady_clock::now();

        auto t0 = std::chrono::steady_clock::now();
        int rc = llama_decode(ctx_, batch);
        auto t1 = std::chrono::steady_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1 - t0).count();

        // Emit model-wide LayerLatency packet
        {
            TelemetryPacket p = make_packet(PacketKind::LayerLatency, ++seq_counter_);
            p.layer_id   = -1;
            p.layer_type = LayerType::Unknown;
            p.latency_us = us;
            dispatch(p, sink_, detector_);
        }

        // Emit per-layer LayerLatency packets
        for (int i = 0; i < topology_.n_layer; ++i) {
            TelemetryPacket p = make_packet(PacketKind::LayerLatency, ++seq_counter_);
            p.layer_id   = i;
            if (model_ && i < static_cast<int>(model_->layers.size())) {
                p.layer_type = deduce_layer_type(model_->layers[i]);
            } else {
                p.layer_type = LayerType::AttentionSelf;
            }
            p.latency_us = layer_latencies_us_[i];
            dispatch(p, sink_, detector_);
        }

        if (rc != 0) {
            log->error("llama_decode returned {}", rc);
            TelemetryPacket ap = make_packet(PacketKind::Anomaly, ++seq_counter_);
            ap.layer_id = -1;
            ap.layer_type = LayerType::Token;
            ap.anomaly_code = AnomalyCode::CpuFallback;
            ap.severity = Severity::Error;
            ap.max_abs = static_cast<float>(rc);
            dispatch(ap, sink_, detector_);
            llama_batch_free(batch);
            break;
        }

        const llama_token id = llama_sampler_sample(smpl_, ctx_, -1);
        llama_sampler_accept(smpl_, id);
        
        char piece[128] = {0};
        int  n_piece = llama_token_to_piece(vocab, id, piece, sizeof(piece) - 1, 0, false);
        if (n_piece > 0) {
            piece[n_piece] = '\0';
            log->info("Sampled token id={} text=\"{}\"", id, piece);
            if (on_token) on_token(std::string(piece));
        } else if (n_piece <= 0 && id != llama_token_eos(vocab)) {
            // Unprintable or special token, but not EOS. Show a tiny block.
            if (on_token) on_token("\u2581"); 
        }

        {
            TelemetryPacket p = make_packet(PacketKind::TokenEnd, ++seq_counter_);
            p.layer_id   = -1;
            p.layer_type = LayerType::Token;
            p.latency_us = us;
            p.shape      = {1, 0, 0, 0};
            dispatch(p, sink_, detector_);
        }

        llama_batch_free(batch);

        if (llama_token_is_eog(vocab, id)) {
            break;
        }

        pos += tokens.size();
        tokens.clear();
        tokens.push_back(id);
    }
}

} // namespace llm_tui
