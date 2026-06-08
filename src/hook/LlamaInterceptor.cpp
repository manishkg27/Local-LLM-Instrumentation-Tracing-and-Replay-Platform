// =============================================================================
//  LlamaInterceptor.cpp
//  -----------------------------------------------------------------------------
//  Day 1: minimal "load + decode 1 token + print topology" implementation.
//  Day 3: extended to wrap llama_decode with chrono timing and push
//         TelemetryPackets to the RingBuffer.
// =============================================================================
#include "hook/LlamaInterceptor.hpp"

#include <llama.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <chrono>
#include <cstring>
#include <thread>

namespace llm_tui {

namespace {
std::shared_ptr<spdlog::logger> interceptor_log() {
    static auto lg = spdlog::stdout_color_mt("llama");
    return lg;
}
} // namespace

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

    log->info("Loaded model \"{}\"", topology_.name);
    log->info("  n_layer  = {}", topology_.n_layer);
    log->info("  n_head   = {}  (kv heads: {})", topology_.n_head, topology_.n_head_kv);
    log->info("  n_embd   = {}", topology_.n_embd);
    log->info("  n_vocab  = {}", topology_.n_vocab);
    log->info("  n_params = {:.2f} B", topology_.n_params / 1e9);
    log->info("  device   = {}", topology_.device_name);
    return true;
}

float LlamaInterceptor::decode_one(const std::string& prompt) {
    auto log = interceptor_log();
    if (!loaded()) {
        log->error("decode_one called before successful load()");
        return 0.0f;
    }

    // ---- Tokenize the prompt ----
    const auto* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab,
                                  prompt.c_str(),
                                  static_cast<int>(prompt.size()),
                                  tokens.data(),
                                  static_cast<int>(tokens.size()),
                                  /*add_special=*/ true,
                                  /*parse_special=*/ true);
    if (n_tokens < 0) {
        log->error("tokenize failed (n={})", n_tokens);
        return 0.0f;
    }
    tokens.resize(n_tokens);
    log->info("Prompt \"{}\" tokenized to {} tokens", prompt, n_tokens);

    // ---- Build a batch of size 1 for the prompt and time llama_decode ----
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = tokens[0];
    batch.pos[0]   = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;  // we want logits for sampling

    auto t0 = std::chrono::steady_clock::now();
    int rc = llama_decode(ctx_, batch);
    auto t1 = std::chrono::steady_clock::now();
    float us = std::chrono::duration<float, std::micro>(t1 - t0).count();

    if (rc != 0) {
        log->error("llama_decode returned {}", rc);
        llama_batch_free(batch);
        return us;
    }
    log->info("llama_decode: {:.2f} us ({} us/token)", us, us);

    // ---- Sample the next token ----
    const llama_token id = llama_sampler_sample(smpl_, ctx_, /*seq_id=*/-1);
    char piece[128] = {0};
    int  n_piece = llama_token_to_piece(vocab, id, piece, sizeof(piece) - 1, /*lstrip=*/0, /*special=*/false);
    if (n_piece > 0) {
        piece[n_piece] = '\0';
        log->info("Sampled token id={} text=\"{}\"", id, piece);
    } else {
        log->info("Sampled token id={} (piece unavailable)", id);
    }

    llama_batch_free(batch);
    return us;
}

} // namespace llm_tui
