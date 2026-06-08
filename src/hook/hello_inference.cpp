// =============================================================================
//  hello_inference.cpp
//  -----------------------------------------------------------------------------
//  Day 1 — Person 1 deliverable.
//
//  Loads a GGUF model via LlamaInterceptor, decodes 1 token, prints layer
//  topology and decode latency. Verifies that the entire stack (vcpkg +
//  ftxui/spdlog + llama.cpp shared libs) builds and runs end-to-end.
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
    log->info("=== hello_inference (Day 1) ===");

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

    // ---- Hook + load ----
    llm_tui::LlamaInterceptor hook;
    if (!hook.load(model_path, /*n_ctx=*/2048, /*n_threads=*/0)) {
        log->error("LlamaInterceptor::load() failed");
        return 3;
    }

    // ---- Decode a tiny prompt to exercise the path ----
    try {
        const std::string prompt = "Hello";
        float us = hook.decode_one(prompt);
        log->info("decode_one() took {:.2f} us total", us);

        // ---- Summary ----
        const auto& t = hook.topology();
        std::cout << "\n"
                  << "╔══════════════════════════════════════════════════════════╗\n"
                  << "║           LLM-TUI Day 1 — hello_inference PASS          ║\n"
                  << "╠══════════════════════════════════════════════════════════╣\n"
                  << "║  Model   : " << t.name << "\n"
                  << "║  n_layer : " << t.n_layer << "\n"
                  << "║  n_head  : " << t.n_head << "  (kv heads: " << t.n_head_kv << ")\n"
                  << "║  n_embd  : " << t.n_embd << "\n"
                  << "║  n_vocab : " << t.n_vocab << "\n"
                  << "║  n_params: " << (t.n_params / 1e9) << " B\n"
                  << "║  device  : " << t.device_name << "\n"
                  << "║  decode  : " << us << " us\n"
                  << "╚══════════════════════════════════════════════════════════╝\n";
    } catch (const std::exception& e) {
        log->error("decode_one threw: {}", e.what());
        return 4;
    }

    log->info("Done.");
    return 0;
}
