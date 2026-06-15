// =============================================================================
//  LlamaInterceptor.cpp
// =============================================================================
#include "hook/LlamaInterceptor.hpp"
#include "llama-model.h"
#include <ggml-backend.h>

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

LlamaInterceptor::LlamaInterceptor() {
    metrics_worker_ = std::thread(&LlamaInterceptor::metrics_worker_loop, this);
}

LlamaInterceptor::~LlamaInterceptor() {
    worker_running_ = false;
    if (metrics_worker_.joinable()) {
        metrics_worker_.join();
    }
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
        bool is_qcur = base_name.find("Qcur") != std::string::npos || base_name.find("attn_q") != std::string::npos;
        bool is_kqv = base_name.find("kqv_out") != std::string::npos || base_name.find("attn_kqv") != std::string::npos || base_name.find("attn_v") != std::string::npos;
        bool is_ffn = base_name.find("ffn_down") != std::string::npos || base_name.find("ffn_gate") != std::string::npos || base_name.find("ffn_up") != std::string::npos || (base_name.find("ffn") != std::string::npos && base_name.find("norm") == std::string::npos);
        bool is_norm = base_name.find("norm") != std::string::npos;
        bool is_kq_soft_max = base_name.find("kq_soft_max") != std::string::npos || base_name.find("attn_kq_soft_max") != std::string::npos;

        if (is_qcur || is_kqv || is_ffn || is_norm || is_kq_soft_max) {
            int64_t n = ggml_nelements(t);
            if (n > 0) {
                // Limit sampling size to 4096 elements to keep timing overhead negligible
                int64_t step = 1;
                int64_t limit = n;
                if (n > 4096) {
                    step = n / 4096;
                    limit = 4096 * step;
                }
                int64_t counted = limit / step;

                // Copy data safely from device/host to our sample buffer
                std::vector<float> sample(counted);
                if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
                    const float* data = (const float*)t->data;
                    for (int64_t i = 0, j = 0; i < limit; i += step, ++j) {
                        sample[j] = data[i];
                    }
                } else if (t->buffer) {
                    // GPU / Device tensor
                    for (int64_t i = 0, j = 0; i < limit; i += step, ++j) {
                        ggml_backend_tensor_get(t, &sample[j], i * sizeof(float), sizeof(float));
                    }
                } else {
                    const float* data = (const float*)t->data;
                    for (int64_t i = 0, j = 0; i < limit; i += step, ++j) {
                        sample[j] = data[i];
                    }
                }

                TelemetryPacket p = make_packet(PacketKind::TensorStats, ++seq_counter_);
                p.layer_id = layer_id;
                
                bool is_kq_soft_max = (base_name == "kq_soft_max" || base_name.find("kq_soft_max") != std::string::npos);
                
                if (base_name == "Qcur" || base_name == "Kcur" || base_name == "kqv_out" || is_kq_soft_max ||
                    base_name.find("attn") != std::string::npos) {
                    p.layer_type = LayerType::AttentionSelf;
                } else if (base_name.find("ffn_down") != std::string::npos || 
                           base_name.find("ffn_gate") != std::string::npos || 
                           base_name.find("ffn_up") != std::string::npos ||
                           base_name.find("ffn") != std::string::npos && base_name.find("norm") == std::string::npos) {
                    p.layer_type = LayerType::Mlp;
                } else if (base_name == "norm" || base_name.find("attn_norm") != std::string::npos ||
                           base_name.find("ffn_norm") != std::string::npos) {
                    p.layer_type = LayerType::RmsNorm;
                }
                
                p.shape = {
                    static_cast<int32_t>(t->ne[0]),
                    static_cast<int32_t>(t->ne[1]),
                    static_cast<int32_t>(t->ne[2]),
                    static_cast<int32_t>(t->ne[3])
                };
                
                p.device = (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) ? 1 : 0; // 1=CUDA/Device, 0=CPU
                
                // -----------------------------------------------------------------------
                // Extract attention matrix from kq_soft_max activations.
                //
                // The ggml tensor shape for kq_soft_max is [n_kv, n_head, n_tokens]:
                //   ne[0] = n_kv  (number of key/value positions in cache)
                //   ne[1] = n_head (number of attention heads)
                //   ne[2] = n_tokens (batch dimension for autoregressive decoding)
                //
                // Memory layout: for head h, query position t, key position k:
                //   data[(h + t * n_head) * n_kv + k]
                //
                // We extract a 7x7 patch from head 0, showing the last 7 query
                // positions attending to the last 7 key positions.
                // -----------------------------------------------------------------------
                if (is_kq_soft_max && t->ne[0] > 0 && t->ne[1] > 0 && t->ne[2] > 0) {
                    const int64_t n_kv   = t->ne[0];  // key/value cache length
                    const int64_t n_head = t->ne[1];  // number of attention heads
                    const int64_t n_tok  = t->ne[2];  // number of query tokens

                    // Use active_head for visualization
                    int head = 0;
                    if (active_head_ptr_) {
                        head = active_head_ptr_->load();
                    }
                    head = std::clamp(head, 0, static_cast<int>(n_head - 1));
                    p.head_idx = static_cast<int8_t>(head);
                    p.attn_seq_len = static_cast<int8_t>(std::min(n_tok, static_cast<int64_t>(127)));

                    // We want to extract the last min(7, n_tok) query positions
                    // and for each, the last min(7, n_kv) key positions.
                    const int view_rows = static_cast<int>(std::min(static_cast<int64_t>(7), n_tok));
                    const int view_cols = static_cast<int>(std::min(static_cast<int64_t>(7), n_kv));

                    const int row_start = static_cast<int>(n_tok - view_rows);
                    const int col_start = static_cast<int>(n_kv - view_cols);

                    float patch_max = 0.0f;

                    // We fetched samples above but for attention patch, we need full patch.
                    // Instead of full sync which is slow, just get patch.
                    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
                        for (int r = 0; r < view_rows; ++r) {
                            for (int c = 0; c < view_cols; ++c) {
                                int64_t offset = (head + (row_start + r) * n_head) * n_kv + (col_start + c);
                                float val = 0.0f;
                                ggml_backend_tensor_get(t, &val, offset * sizeof(float), sizeof(float));
                                val = std::abs(val);
                                if (val > patch_max) patch_max = val;
                            }
                        }
                    } else {
                        const float* data = (const float*)t->data;
                        for (int r = 0; r < view_rows; ++r) {
                            for (int c = 0; c < view_cols; ++c) {
                                int64_t offset = (head + (row_start + r) * n_head) * n_kv + (col_start + c);
                                float val = std::abs(data[offset]);
                                if (val > patch_max) patch_max = val;
                            }
                        }
                    }
                    if (patch_max < 1e-12f) patch_max = 1.0f;

                    // Zero out the patch buffer
                    std::fill(std::begin(p.attn_patch), std::end(p.attn_patch), 0.0f);

                    // Place the 7x7 data in the buffer, aligned to bottom-right
                    const int patch_r_start = 7 - view_rows;
                    const int patch_c_start = 7 - view_cols;

                    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
                        for (int r = 0; r < view_rows; ++r) {
                            for (int c = 0; c < view_cols; ++c) {
                                int64_t offset = (head + (row_start + r) * n_head) * n_kv + (col_start + c);
                                float val = 0.0f;
                                ggml_backend_tensor_get(t, &val, offset * sizeof(float), sizeof(float));
                                val = std::abs(val) / patch_max;
                                p.attn_patch[(patch_r_start + r) * 7 + (patch_c_start + c)] = val;
                            }
                        }
                    } else {
                        const float* data = (const float*)t->data;
                        for (int r = 0; r < view_rows; ++r) {
                            for (int c = 0; c < view_cols; ++c) {
                                int64_t offset = (head + (row_start + r) * n_head) * n_kv + (col_start + c);
                                float val = std::abs(data[offset]) / patch_max;
                                p.attn_patch[(patch_r_start + r) * 7 + (patch_c_start + c)] = val;
                            }
                        }
                    }

                    interceptor_log()->debug(
                        "kq_soft_max layer={} n_kv={} n_head={} n_tok={} patch_max={:.6f}",
                        layer_id, n_kv, n_head, n_tok, patch_max);

                    // -----------------------------------------------------------------
                    // Side-channel: also store the full attention matrix for the
                    // TUI to read. We only store one snapshot at a time (the most
                    // recent kq_soft_max for the active layer/head).
                    // -----------------------------------------------------------------
                    {
                        AttentionMatrix full_matrix;
                        full_matrix.layer_id = layer_id;
                        full_matrix.head_id = 0;
                        full_matrix.n_kv = static_cast<int>(n_kv);
                        full_matrix.n_head = static_cast<int>(n_head);
                        int max_dim = 256;
                        int copy_tok = std::min(static_cast<int>(n_tok), max_dim);
                        int copy_kv  = std::min(static_cast<int>(n_kv), max_dim);

                        full_matrix.n_tok = copy_tok;
                        full_matrix.n_kv = copy_kv;
                        full_matrix.timestamp_ns = p.timestamp_ns;
                        
                        // We extract the *last* copy_tok query positions and *last* copy_kv key positions
                        int64_t start_tok = n_tok - copy_tok;
                        int64_t start_kv = n_kv - copy_kv;

                        // Store the full matrix for active head
                        full_matrix.data.resize(static_cast<size_t>(copy_tok) * copy_kv);
                        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
                            // Extract full matrix from GPU
                            for (int64_t t_i = 0; t_i < copy_tok; ++t_i) {
                                for (int64_t k_i = 0; k_i < copy_kv; ++k_i) {
                                    int64_t offset = (head + (start_tok + t_i) * n_head) * n_kv + (start_kv + k_i);
                                    float val = 0.0f;
                                    ggml_backend_tensor_get(t, &val, offset * sizeof(float), sizeof(float));
                                    full_matrix.data[t_i * copy_kv + k_i] = std::abs(val) / patch_max;
                                }
                            }
                        } else {
                            const float* data = (const float*)t->data;
                            for (int64_t t_i = 0; t_i < copy_tok; ++t_i) {
                                for (int64_t k_i = 0; k_i < copy_kv; ++k_i) {
                                    int64_t offset = (head + (start_tok + t_i) * n_head) * n_kv + (start_kv + k_i);
                                    full_matrix.data[t_i * copy_kv + k_i] = std::abs(data[offset]) / patch_max;
                                }
                            }
                        }
                        attn_cache_.store(std::move(full_matrix));
                    }
                }
                
                PendingMetrics pm;
                pm.p = p;
                pm.data_sample = std::move(sample);
                metrics_queue_.push(pm);
            }
        }
    }

    return true;
}

void LlamaInterceptor::metrics_worker_loop() {
    while (worker_running_) {
        if (auto opt = metrics_queue_.pop_for(std::chrono::milliseconds(10))) {
            auto pm = *opt;
            double sum = 0.0;
            double sum_sq = 0.0;
            float max_abs = 0.0f;
            int64_t zero_count = 0;
            int64_t counted = pm.data_sample.size();

            if (counted > 0) {
                for (float val : pm.data_sample) {
                    sum += val;
                    sum_sq += val * val;
                    float abs_val = std::abs(val);
                    if (abs_val > max_abs) max_abs = abs_val;
                    if (abs_val < 1e-6f) zero_count++;
                }

                float mean = static_cast<float>(sum / counted);
                float variance = static_cast<float>((sum_sq / counted) - (mean * mean));
                pm.p.sigma = std::sqrt(std::max(0.0f, variance));
                pm.p.sparsity = static_cast<float>(zero_count) / counted;
                pm.p.mean = mean;
                pm.p.max_abs = max_abs;
            }

            dispatch(pm.p, sink_, detector_);
        }
    }
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
        } else if (n_piece <= 0 && id != llama_vocab_eos(vocab)) {
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

        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        pos += tokens.size();
        tokens.clear();
        tokens.push_back(id);
    }
}

} // namespace llm_tui
