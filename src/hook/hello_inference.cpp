// =============================================================================
//  hello_inference.cpp
//  -----------------------------------------------------------------------------
//  Day 1 + Day 2 — Person 1 deliverable.
//
//  Loads a GGUF model via LlamaInterceptor, decodes 1 token, prints layer
//  topology and decode latency, drains the telemetry ring buffer, and
//  shows the first end-to-end data path: Interceptor -> RingBuffer ->
//  AnomalyDetector.
//
//  Usage:
//      build/hello_inference [path-to-gguf]
//  Defaults to /home/manish/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
// =============================================================================
#include "hook/LlamaInterceptor.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace {
std::shared_ptr<spdlog::logger> app_log() {
    static auto lg = spdlog::stdout_color_mt("app");
    lg->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n] %v");
    lg->set_level(spdlog::level::info);
    return lg;
}
} // namespace

int main(int argc, char** argv) {
    auto log = app_log();
    log->info("=== hello_inference (Day 2) ===");

    // ---- Resolve model path ----
    std::string model_path =
        (argc > 1) ? std::string(argv[1])
                   : "/home/manish/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf";

    // Allow override via env var
    if (const char* env = std::getenv("LLM_TUI_MODEL"); env && *env) {
        model_path = env;
    }

    if (!fs::exists(model_path)) {
        log->error("Model file not found: {}", model_path);
        log->error("Download with:");
        log->error("  wget -c https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf");
        return 2;
    }
    log->info("Model: {}", model_path);
    log->info("Model size: {:.2f} MB", fs::file_size(model_path) / (1024.0 * 1024.0));

    // ---- Wire a sink so the TUI consumer (D6) can drain the same queue ----
    // For Day 2 we just allocate a small buffer and drain it after the
    // decode call to prove the funnel works end-to-end.
    llm_tui::RingBuffer<llm_tui::TelemetryPacket> sink(512);

    // ---- Hook + load ----
    llm_tui::LlamaInterceptor hook;
    hook.set_sink(&sink);
    if (!hook.load(model_path, /*n_ctx=*/2048, /*n_threads=*/0)) {
        log->error("LlamaInterceptor::load() failed");
        return 3;
    }

    // ---- Decode a tiny prompt to exercise the path ----
    try {
        const std::vector<std::string> prompts = {"Hello"};
        for (const auto& prompt : prompts) {
            fmt::print("Prompt: \"{}\"\n", prompt);
            
            hook.generate(prompt, [](const std::string& text) {
                fmt::print("{}", text);
                fflush(stdout);
            });
            fmt::print("\n\n");
        }

        // ---- Drain the sink and show what came through ----
        std::vector<llm_tui::TelemetryPacket> drained;
        while (true) {
            auto p = sink.pop_for(std::chrono::milliseconds(5));
            if (!p.has_value()) break;
            drained.push_back(*p);
        }
        log->info("Drained {} telemetry packets from sink", drained.size());
        for (auto& p : drained) {
            log->info("  [seq={:>4}] kind={:<14} layer={:>3}  lat={:.1f}us",
                      p.sequence_id,
                      llm_tui::to_string(p.kind),
                      p.layer_id,
                      p.latency_us);
        }

        // ---- AnomalyDetector summary ----
        const auto lg2 = hook.detector().ledger();
        log->info("AnomalyDetector: {} entries in ledger", lg2.size());

        // ---- Summary box ----
        const auto& t = hook.topology();
        std::cout << "\n"
                  << "╔══════════════════════════════════════════════════════════╗\n"
                  << "║           LLM-TUI Day 2 — hello_inference PASS          ║\n"
                  << "╠══════════════════════════════════════════════════════════╣\n"
                  << "║  Model   : " << t.name << "\n"
                  << "║  n_layer : " << t.n_layer << "\n"
                  << "║  n_head  : " << t.n_head << "  (kv heads: " << t.n_head_kv << ")\n"
                  << "║  n_embd  : " << t.n_embd << "\n"
                  << "║  n_vocab : " << t.n_vocab << "\n"
                  << "║  n_params: " << (t.n_params / 1e9) << " B\n"
                  << "║  device  : " << t.device_name << "\n"
                  << "║  packets : " << drained.size() << " (drained from sink)\n"
                  << "║  anomaly : " << lg2.size() << " (in detector ledger)\n"
                  << "╚══════════════════════════════════════════════════════════╝\n";
    } catch (const std::exception& e) {
        log->error("decode_one threw: {}", e.what());
        return 4;
    }

    log->info("Done.");
    return 0;
}
