// =============================================================================
//  tui_hello.cpp
//  -----------------------------------------------------------------------------
//  Day 3 — Person 2 deliverable.
//
//  Refactored to be fully modular and interactive:
//    - Instantiates AppState, LlamaInterceptor, and RingBuffer.
//    - Runs the real model in a background thread if the model file is found,
//      otherwise runs a high-fidelity simulator to demonstrate the telemetry flow.
//    - Sets up a consumer thread to drain the RingBuffer and update AppState.
//    - Imports the modular Panel components (1 to 5) and lays them out.
//    - Handles smart focus navigation (Arrow keys/hjkl bubble up to move focus).
// =============================================================================
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>
#include <ftxui/screen/terminal.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>

#include "tui/AppState.hpp"
#include "tui/PanelTopology.hpp"
#include "tui/PanelPacketStream.hpp"
#include "tui/PanelAttention.hpp"
#include "tui/PanelMetrics.hpp"
#include "tui/PanelAnomalies.hpp"

#include "hook/LlamaInterceptor.hpp"
#include "core/RingBuffer.hpp"
#include "core/TelemetryPacket.hpp"
#include "core/AnomalyDetector.hpp"

namespace {
std::shared_ptr<spdlog::logger> tui_log() {
    static auto lg = spdlog::stdout_color_mt("tui");
    lg->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n] %v");
    return lg;
}

// ASCII art header
ftxui::Element header() {
    using namespace ftxui;
    auto title = "  Local LLM Instrumentation, Tracing & Replay Platform  ";
    ftxui::Elements rows = {
        text("╔════════════════════════════════════════════════════════════════════╗") | color(Color::Yellow),
        text(std::string("║") + title + "║") | bold | color(Color::Yellow),
        text("╚════════════════════════════════════════════════════════════════════╝") | color(Color::Yellow),
    };
    return vbox(std::move(rows)) | center;
}

ftxui::Element footer(int keypresses) {
    using namespace ftxui;
    ftxui::Elements rows = {
        text(" [") | color(Color::GrayDark),
        text("Tab") | color(Color::GreenLight) | bold,
        text("] cycle  [") | color(Color::GrayDark),
        text("h/j/k/l or Arrows") | color(Color::GreenLight) | bold,
        text("] move focus / navigate  [") | color(Color::GrayDark),
        text("Space/Enter") | color(Color::GreenLight) | bold,
        text("] expand/collapse  [") | color(Color::GrayDark),
        text("q") | color(Color::RedLight) | bold,
        text("] quit   keypresses=") | color(Color::GrayDark),
        text(std::to_string(keypresses)) | color(Color::Cyan),
        text("]") | color(Color::GrayDark),
    };
    return hbox(std::move(rows)) | center;
}

// Custom status panel component
class StatusPanelBase : public ftxui::ComponentBase {
    std::weak_ptr<llm_tui::AppState> state_;
    bool is_live_;
public:
    StatusPanelBase(std::weak_ptr<llm_tui::AppState> state, bool is_live)
        : state_(state), is_live_(is_live) {}
        
    ftxui::Element OnRender() override {
        using namespace ftxui;
        auto state = state_.lock();
        if (!state) {
            return text("No State") | dim;
        }
        std::lock_guard<std::recursive_mutex> lock(state->mutex);
        int total_packets = static_cast<int>(state->packets.size());
        int total_anomalies = static_cast<int>(state->anomalies.size());
        
        return vbox(Elements{
            hbox(Elements{text("Status:  ") | dim, text("ACTIVE") | color(Color::Green) | bold}),
            hbox(Elements{text("Mode:    ") | dim, text(is_live_ ? "LIVE LLM" : "SIMULATOR") | color(Color::Cyan)}),
            hbox(Elements{text("Packets: ") | dim, text(std::to_string(total_packets))}),
            hbox(Elements{text("Alerts:  ") | dim, text(std::to_string(total_anomalies)) | color(total_anomalies > 0 ? Color::Red : Color::Green)}),
            separator(),
            text("Prompt: " + state->current_prompt) | dim,
            vbox({
                paragraph(state->generated_text) | color(Color::White)
            }) | yframe | vscroll_indicator | flex,
        }) | flex;
    }
    bool Focusable() const override { return true; }
};

} // namespace

int main(int argc, char** argv) {
    using namespace ftxui;
    tui_log()->info("llm-tui TUI starting...");

    // ---- Resolve model path ----
    std::string model_path =
        (argc > 1) ? std::string(argv[1])
                   : "/home/manish/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf";

    if (const char* env = std::getenv("LLM_TUI_MODEL"); env && *env) {
        model_path = env;
    }

    bool is_live = std::filesystem::exists(model_path);
    tui_log()->info("Model path: {} (exists={})", model_path, is_live);

    // ---- Shared TUI state & queue -------------------------------------
    auto state = std::make_shared<llm_tui::AppState>();
    llm_tui::RingBuffer<llm_tui::TelemetryPacket> sink(1024);
    llm_tui::LlamaInterceptor hook;
    hook.detector().set_config({.log_to_stderr = false});
    hook.set_sink(&sink);

    std::atomic<bool> running{true};
    std::atomic<int> keypresses{0};
    
    // Command bar state
    bool command_active = false;
    std::string command_input = "";
    
    // Prompt queue
    std::mutex prompt_mutex;
    std::vector<std::string> prompt_queue = {
        "Explain virtual functions in C++.",
        "How does self-attention work?",
        "Write a quicksort in python.",
    };

    // Default prompts for continuous streaming
    std::vector<std::string> default_prompts = {
        "Explain virtual functions in C++.",
        "How does self-attention work?",
        "Write a quicksort in python.",
    };
    int default_idx = 0;

    // ---- Background worker thread: runs real model or simulator -------
    std::thread worker;
    if (is_live) {
        worker = std::thread([&, model_path, default_prompts, default_idx]() mutable {
            if (hook.load(model_path, /*n_ctx=*/2048, /*n_threads=*/0)) {
                while (running.load()) {
                    std::string prompt_str;
                    {
                        std::lock_guard<std::mutex> lk(prompt_mutex);
                        if (!prompt_queue.empty()) {
                            prompt_str = prompt_queue.front();
                            prompt_queue.erase(prompt_queue.begin());
                        }
                    }
                    if (!prompt_str.empty()) {
                        {
                            std::lock_guard<std::recursive_mutex> lk(state->mutex);
                            state->current_prompt = prompt_str;
                            state->generated_text = "";
                        }
                        hook.generate(prompt_str, [&](const std::string& piece) {
                            std::lock_guard<std::recursive_mutex> lk(state->mutex);
                            state->generated_text += piece;
                        });
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    } else {
                        {
                            std::lock_guard<std::recursive_mutex> lk(state->mutex);
                            state->current_prompt = default_prompts[default_idx];
                            state->generated_text = "";
                        }
                        hook.generate(default_prompts[default_idx], [&](const std::string& piece) {
                            std::lock_guard<std::recursive_mutex> lk(state->mutex);
                            state->generated_text += piece;
                        });
                        default_idx = (default_idx + 1) % default_prompts.size();
                        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    }
                }
            } else {
                tui_log()->error("Failed to load model in background thread.");
            }
        });
    } else {
        // High-fidelity simulator to demonstrate the TUI panels
        worker = std::thread([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            // Emit model topology
            {
                llm_tui::TelemetryPacket p = llm_tui::make_packet(llm_tui::PacketKind::Topology, 1);
                p.layer_id = -1;
                p.shape = {36, 16, 2048, 151936};
                p.device = 0;
                p.sigma = 3.09f; // 3.09 B params
                sink.push(p);
            }
            // Emit layer topologies
            for (int i = 0; i < 36; ++i) {
                llm_tui::TelemetryPacket lp = llm_tui::make_packet(llm_tui::PacketKind::Topology, i + 2);
                lp.layer_id = i;
                lp.layer_type = (i % 2 == 0) ? llm_tui::LayerType::AttentionSelf : llm_tui::LayerType::Mlp;
                lp.shape = {16, 8, 2048, 8192};
                sink.push(lp);
            }

            uint32_t seq = 40;
            while (running.load()) {
                sink.push(llm_tui::make_packet(llm_tui::PacketKind::TokenStart, ++seq));
                for (int i = 0; i < 36; ++i) {
                    if (!running.load()) break;

                    llm_tui::TelemetryPacket lp = llm_tui::make_packet(llm_tui::PacketKind::LayerLatency, ++seq);
                    lp.layer_id = i;
                    lp.latency_us = 800.0f + (rand() % 400);
                    sink.push(lp);

                    if (rand() % 4 == 0) {
                        llm_tui::TelemetryPacket ts = llm_tui::make_packet(llm_tui::PacketKind::TensorStats, ++seq);
                        ts.layer_id = i;
                        ts.sparsity = (rand() % 100) / 100.0f;
                        ts.mean = (rand() % 100) / 20000.0f;
                        ts.max_abs = 2.0f + (rand() % 100) / 15.0f;
                        ts.shape = {1, 1, 16, 128};
                        sink.push(ts);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
                }
                
                llm_tui::TelemetryPacket p = llm_tui::make_packet(llm_tui::PacketKind::TokenEnd, ++seq);
                p.latency_us = 45000.0f;
                sink.push(p);

                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            }
        });
    }

    // ---- Background consumer thread: drains queue to AppState ---------
    std::thread consumer([&] {
        while (running.load()) {
            auto first = sink.pop_for(std::chrono::milliseconds(10));
            if (first.has_value()) {
                std::vector<llm_tui::TelemetryPacket> batch;
                batch.push_back(std::move(*first));
                
                // Drain any additional packets that are currently in the queue
                while (auto next = sink.try_pop()) {
                    batch.push_back(std::move(*next));
                }
                
                // Process the batch (update_from_packet locks state->mutex internally)
                for (const auto& p : batch) {
                    state->update_from_packet(p);
                }
                
                // Update the anomalies ledger once per batch
                if (is_live) {
                    auto ledger = hook.detector().ledger();
                    std::lock_guard<std::recursive_mutex> lock(state->mutex);
                    state->anomalies = std::move(ledger);
                } else {
                    // Simulate anomalies in simulator mode
                    std::lock_guard<std::recursive_mutex> lock(state->mutex);
                    for (const auto& p : batch) {
                        if (p.kind == llm_tui::PacketKind::TensorStats && p.max_abs > 6.0f && state->anomalies.size() < 100) {
                            llm_tui::LedgerEntry le;
                            le.severity = (p.max_abs > 7.5f) ? llm_tui::Severity::Error : llm_tui::Severity::Warn;
                            le.code = llm_tui::AnomalyCode::OutlierFeature;
                            le.layer_id = p.layer_id;
                            le.timestamp_ns = p.timestamp_ns;
                            le.message = "layer " + std::to_string(p.layer_id) + ": |x|_max=" + std::to_string(p.max_abs).substr(0, 5) + " > outlier threshold";
                            state->anomalies.push_back(le);
                        }
                    }
                }
            }
        }
    });

    // ---- Instantiate modular panels -----------------------------------
    auto panel1 = llm_tui::CreatePanelTopology(state);
    auto panel2 = llm_tui::CreatePanelPacketStream(state);
    auto panel3 = llm_tui::CreatePanelAttention(state);
    auto panel4 = llm_tui::CreatePanelMetrics(state);
    auto panel5 = llm_tui::CreatePanelAnomalies(state);
    auto status_panel = std::make_shared<StatusPanelBase>(state, is_live);

    // Decorator helper to draw windows around panels
    auto make_panel = [](const std::string& title, Component comp) {
        return Renderer(comp, [title, comp] {
            auto win = window(text(title) | bold, comp->Render());
            if (comp->Focused()) {
                return win | color(Color::Yellow);
            } else {
                return win | color(Color::GrayDark);
            }
        });
    };

    auto p1 = make_panel(" 1. MODEL TOPOLOGY ", panel1);
    auto p2 = make_panel(" 2. LIVE PACKET STREAM ", panel2);
    auto p3 = make_panel(" 3. ATTENTION MATRIX ", panel3);
    auto p4 = make_panel(" 4. RUNTIME METRICS ", panel4);
    auto p5 = make_panel(" 5. ANOMALY LEDGER ", panel5);
    auto p6 = make_panel(" 6. LLM OUTPUT / STATUS ", status_panel);

    // ---- Container layouts --------------------------------------------
    auto row1 = Container::Horizontal({p1, p2});
    auto row2 = Container::Horizontal({p3});
    auto row3 = Container::Horizontal({p4, p5, p6});
    auto main_container = Container::Vertical({row1, row2, row3});

    // ---- Main Renderer ------------------------------------------------
    auto main_renderer = Renderer(main_container, [&] {
        std::lock_guard<std::recursive_mutex> lock(state->mutex);
        
        auto term_size = Terminal::Size();
        if (term_size.dimx < 80 || term_size.dimy < 20) {
            return vbox(Elements{
                filler(),
                text(" ⚠ Terminal too small ⚠ ") | bold | color(Color::Red) | center,
                text(fmt::format("Current size: {}x{}, Required size: >= 80x20", term_size.dimx, term_size.dimy)) | center,
                text("Please enlarge your terminal window to resume layout.") | dim | center,
                filler()
            });
        }
        
        Element footer_el;
        if (command_active) {
            footer_el = hbox(Elements{
                text(":") | bold | color(Color::Yellow),
                text(command_input) | bold | color(Color::White),
                text("█") | color(Color::Yellow)
            });
        } else {
            footer_el = hbox(Elements{
                text(" Msg: ") | dim,
                text(state->status_message) | color(Color::Cyan),
                filler(),
                text("[Tab] cycle  [:] command  [?] help  [q] quit ") | dim
            });
        }

        if (state->attention_fullscreen) {
            return p3->Render() | flex_grow;
        }
        
        return vbox(Elements{
            header(),
            separator(),
            hbox(Elements{
                p1->Render() | size(WIDTH, LESS_THAN, 50) | flex,
                separator(),
                p2->Render() | flex_grow,
            }) | flex_grow,
            separator(),
            p3->Render() | flex_grow,
            separator(),
            hbox(Elements{
                p4->Render() | flex_grow,
                separator(),
                p5->Render() | flex_grow,
                separator(),
                p6->Render() | flex_grow,
            }) | flex_grow,
            separator(),
            footer_el
        });
    });

    // ---- Event handler — smart focus bubble navigation ----------------
    auto component = CatchEvent(main_renderer, [&](Event evt) {
        // If in command mode, capture keyboard input and block all other keys
        if (command_active) {
            if (evt == Event::Escape) {
                command_active = false;
                command_input = "";
                std::lock_guard<std::recursive_mutex> lock(state->mutex);
                state->status_message = "Command cancelled.";
                return true;
            }
            if (evt == Event::Return) {
                std::string cmd = command_input;
                command_active = false;
                command_input = "";
                
                std::lock_guard<std::recursive_mutex> lock(state->mutex);
                if (cmd == "q" || cmd == "quit") {
                    running = false;
                    state->status_message = "Exiting...";
                } else if (cmd == "help" || cmd == "?" || cmd == "h") {
                    state->status_message = "Commands: :q (quit) | :p <text> (prompt) | :c <val> (contrast) | :help";
                } else if (cmd.rfind("p ", 0) == 0 || cmd.rfind("prompt ", 0) == 0) {
                    size_t space_idx = cmd.find(' ');
                    if (space_idx != std::string::npos) {
                        std::string p_str = cmd.substr(space_idx + 1);
                        {
                            std::lock_guard<std::mutex> lk(prompt_mutex);
                            prompt_queue.push_back(p_str);
                        }
                        state->status_message = "Prompt queued: " + p_str;
                    }
                } else if (cmd.rfind("c ", 0) == 0 || cmd.rfind("contrast ", 0) == 0) {
                    size_t space_idx = cmd.find(' ');
                    if (space_idx != std::string::npos) {
                        try {
                            float c_val = std::stof(cmd.substr(space_idx + 1));
                            state->attention_contrast = std::clamp(c_val, 0.1f, 5.0f);
                            state->status_message = "Attention contrast set to " + std::to_string(state->attention_contrast).substr(0, 4);
                        } catch (...) {
                            state->status_message = "Error: Invalid float value.";
                        }
                    }
                } else {
                    state->status_message = "Unknown command: :" + cmd;
                }
                return true;
            }
            if (evt == Event::Backspace) {
                if (!command_input.empty()) {
                    command_input.pop_back();
                }
                return true;
            }
            if (evt.is_character()) {
                command_input += evt.character();
                return true;
            }
            
            // Block all other keys (like arrow keys) while typing a command
            return true;
        }

        // 1. Give children the first opportunity to handle the event
        if (main_renderer->OnEvent(evt)) {
            return true;
        }

        // 2. If children didn't handle it, process globally or for navigation
        
        // Trigger command mode
        if (evt == Event::Character(':')) {
            command_active = true;
            command_input = "";
            return true;
        }
        
        // Show help message
        if (evt == Event::Character('?')) {
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            state->status_message = "Commands: :q (quit) | :p <text> (prompt) | :c <val> (contrast) | :help";
            return true;
        }

        // Esc exits fullscreen mode first if active
        if (evt == Event::Escape) {
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            if (state->attention_fullscreen) {
                state->attention_fullscreen = false;
                return true;
            }
        }

        // q or Esc to quit
        if (evt == Event::Character('q') || evt == Event::Escape) {
            running = false;
            return true;
        }

        // Keep track of total keypresses
        keypresses++;

        // Determine which panel is currently focused
        int current_idx = 0;
        std::vector<Component> focusable_panels = {p1, p2, p3, p4, p5, p6};
        for (size_t i = 0; i < focusable_panels.size(); ++i) {
            if (focusable_panels[i]->Focused()) {
                current_idx = i;
                break;
            }
        }

        // Tab focus cycling
        if (evt == Event::Tab) {
            current_idx = (current_idx + 1) % focusable_panels.size();
            focusable_panels[current_idx]->TakeFocus();
            return true;
        }
        if (evt == Event::TabReverse) {
            current_idx = (current_idx + focusable_panels.size() - 1) % focusable_panels.size();
            focusable_panels[current_idx]->TakeFocus();
            return true;
        }

        return false;
    });

    // ---- FTXUI Interactive Loop ---------------------------------------
    auto screen = ScreenInteractive::TerminalOutput();

    // Redraw loop at ~30 FPS
    std::thread ticker([&] {
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(component);

    // ---- Clean shutdown -----------------------------------------------
    running = false;
    sink.close();

    if (ticker.joinable()) ticker.join();
    if (consumer.joinable()) consumer.join();
    if (worker.joinable()) worker.join();

    tui_log()->info("llm-tui TUI exited cleanly (keypresses={})", keypresses.load());
    return 0;
}
