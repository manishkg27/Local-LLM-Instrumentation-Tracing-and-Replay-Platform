// =============================================================================
//  tui_hello.cpp
// =============================================================================
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "tui/AppState.hpp"
#include "tui/PanelTopology.hpp"
#include "tui/PanelPacketStream.hpp"
#include "tui/PanelAttention.hpp"
#include "tui/PanelMetrics.hpp"
#include "tui/PanelAnomalies.hpp"
#include "hook/LlamaInterceptor.hpp"

namespace {
std::shared_ptr<spdlog::logger> tui_log() {
    static auto lg = spdlog::stdout_color_mt("tui");
    lg->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n] %v");
    return lg;
}

ftxui::Element stat_box(const std::string& label, std::string value) {
    using namespace ftxui;
    if (value.size() > 300) {
        value = "..." + value.substr(value.size() - 300);
    }
    ftxui::Elements rows = {
        text(label) | bold,
        separatorEmpty(),
        paragraphAlignLeft(value) | color(Color::Cyan),
    };
    return vbox(std::move(rows));
}

// ASCII art header that uses NerdFont box drawing characters
ftxui::Element header() {
    using namespace ftxui;
    auto title = "  Local LLM Instrumentation, Tracing & Replay Platform  ";
    ftxui::Elements rows = {
        text("╔════════════════════════════════════════════════════════════════════╗"),
        text(std::string("║") + title + "║") | bold | color(Color::Yellow),
        text("╚════════════════════════════════════════════════════════════════════╝"),
    };
    return vbox(std::move(rows)) | center;
}

ftxui::Element footer(int keypresses) {
    using namespace ftxui;
    ftxui::Elements rows = {
        text(" [") | color(Color::GrayDark),
        text("h/j/k/l") | color(Color::GreenLight) | bold,
        text("] nav  [") | color(Color::GrayDark),
        text("Tab") | color(Color::GreenLight) | bold,
        text("] cycle  [") | color(Color::GrayDark),
        text("+/-") | color(Color::GreenLight) | bold,
        text("] adjust  [") | color(Color::GrayDark),
        text("q") | color(Color::RedLight) | bold,
        text("] quit   keypresses=") | color(Color::GrayDark),
        text(std::to_string(keypresses)) | color(Color::Cyan),
        text("]") | color(Color::GrayDark),
    };
    return hbox(std::move(rows)) | center;
}

} // namespace

int main() {
    using namespace ftxui;
    using namespace llm_tui;
    tui_log()->info("tui_hello (Day 1) starting");

    auto screen = ScreenInteractive::TerminalOutput();

    // ---- Shared state -------------------------------------------------
    auto state = std::make_shared<AppState>();

    std::atomic<int> keypresses{0};
    std::atomic<bool> running{true};
    int focus_index = 0;

    // ---- Interceptor & Telemetry --------------------------------------
    // Configurable ring buffer size via environment variable
    std::size_t ring_capacity = 1024;
    if (const char* env_cap = std::getenv("LLM_TUI_RING_BUFFER"); env_cap && *env_cap) {
        ring_capacity = static_cast<std::size_t>(std::atoi(env_cap));
        if (ring_capacity < 64) ring_capacity = 64;
    }
    llm_tui::RingBuffer<llm_tui::TelemetryPacket> sink(ring_capacity);
    llm_tui::LlamaInterceptor hook;
    hook.set_sink(&sink);
    hook.set_active_head_ptr(&state->selected_head);
    // Wire the full attention matrix cache from the interceptor to the TUI state
    state->attn_cache_ptr = &hook.attn_cache();

    std::thread inference_thread([&]() {
        std::vector<std::string> prompts = {
            "Hello, what is the meaning of life?",
            "Explain the concept of quantum entanglement in simple terms.",
            "Write a short haiku about a rogue artificial intelligence.",
            "What are the Three Laws of Robotics created by Isaac Asimov?",
            "Can you tell me a short, funny joke about programmers?"
        };
        int prompt_idx = 0;
        
        std::string model_path = "/home/manish/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf";
        if (const char* env = std::getenv("LLM_TUI_MODEL"); env && *env) {
            model_path = env;
        }

        if (hook.load(model_path, 2048, 0)) {
            // continuously generate to keep TUI alive
            while (running.load()) {
                state->current_prompt = "<|im_start|>user\n" + prompts[prompt_idx] + "<|im_end|>\n<|im_start|>assistant\n";
                prompt_idx = (prompt_idx + 1) % prompts.size();
                
                {
                    std::lock_guard<std::recursive_mutex> lock(state->mutex);
                    state->generated_text = "";
                }
                hook.generate(state->current_prompt, [&](const std::string& token) {
                    std::lock_guard<std::recursive_mutex> lock(state->mutex);
                    state->generated_text += token;
                });
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    });

    // ---- Panels -------------------------------------------------------
    std::vector<ftxui::Component> panels = {
        CreatePanelTopology(state),
        CreatePanelPacketStream(state),
        CreatePanelAttention(state),
        CreatePanelMetrics(state),
        CreatePanelAnomalies(state),
        Renderer([state] { 
            std::string st = state->generated_text.empty() ? "Waiting for model..." : state->generated_text;
            return stat_box("GENERATION & STATUS", st); 
        })
    };

    // Decorate panels with borders and highlights
    std::vector<ftxui::Component> decorated_panels;
    std::vector<std::string> titles = {
        "1. MODEL TOPOLOGY", "2. LIVE PACKET STREAM", "3. ATTENTION MATRIX",
        "4. RUNTIME METRICS", "5. ANOMALY LEDGER", "6. STATUS"
    };

    for (size_t i = 0; i < panels.size(); ++i) {
        auto wrapped = Renderer(panels[i], [i, &focus_index, state, &panels, &titles] {
            bool focused = (static_cast<int>(i) == focus_index);
            bool is_fullscreen = (i == 0 && state->topology_fullscreen) || 
                                 (i == 1 && state->packet_stream_fullscreen) || 
                                 (i == 2 && state->attention_fullscreen);
            
            auto content = window(text(titles[i]) | (focused ? bold : dim), panels[i]->Render())
                   | (focused ? borderHeavy : border) | (focused ? color(Color::Yellow) : color(Color::Default));
                   
            if (!is_fullscreen) {
                if (i == 1 || i == 4) {
                    content = content | flex | size(HEIGHT, EQUAL, 24);
                } else if (i == 3) {
                    content = content | size(WIDTH, EQUAL, 55) | size(HEIGHT, EQUAL, 24);
                } else {
                    content = content | size(WIDTH, EQUAL, 40) | size(HEIGHT, EQUAL, 24);
                }
            } else {
                content = content | flex;
            }
            return content;
        });
        decorated_panels.push_back(wrapped);
    }

    // Layout
    auto main_container = Container::Vertical({
        Container::Horizontal({decorated_panels[0], decorated_panels[1], decorated_panels[2]}),
        Container::Horizontal({decorated_panels[3], decorated_panels[4], decorated_panels[5]})
    });

    auto root = Renderer(main_container, [&] {
        ftxui::Element content;
        if (state->topology_fullscreen) {
            content = decorated_panels[0]->Render();
        } else if (state->packet_stream_fullscreen) {
            content = decorated_panels[1]->Render();
        } else if (state->attention_fullscreen) {
            content = decorated_panels[2]->Render();
        } else {
            content = main_container->Render();
        }
        
        auto main_ui = vbox({
            header(),
            separator(),
            content | flex,
            separator(),
            footer(keypresses.load()),
        });
        
        // Wrap in an hbox with flex to force the vbox to consume the full terminal width
        return hbox({ main_ui | flex });
    });

    auto event_handler = CatchEvent(root, [&](Event evt) {
        if (evt == Event::Character('q') || evt == Event::Escape) {
            running = false;
            screen.ExitLoopClosure()();
            return true;
        }

        if (evt == Event::Tab) {
            focus_index = (focus_index + 1) % 6;
            panels[focus_index]->TakeFocus();
            keypresses++;
            return true;
        }





        // Global contrast adjustment
        if (evt == Event::Character('+') || evt == Event::Character('=')) {
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            state->attention_contrast = std::min(5.0f, state->attention_contrast + 0.1f);
            return true;
        }
        if (evt == Event::Character('-') || evt == Event::Character('_')) {
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            state->attention_contrast = std::max(0.1f, state->attention_contrast - 0.1f);
            return true;
        }

        // Let the active panel handle the event first
        if (panels[focus_index]->OnEvent(evt)) {
            keypresses++;
            return true;
        }

        // Cross-panel navigation using arrow keys / hjkl
        if (evt == Event::Character('h') || evt == Event::ArrowLeft) {
            focus_index = (focus_index + 2) % 3 + (focus_index / 3) * 3; // wrap same row left
            panels[focus_index]->TakeFocus();
            keypresses++;
            return true;
        }

        if (evt == Event::Character('l') || evt == Event::ArrowRight) {
            focus_index = (focus_index + 1) % 3 + (focus_index / 3) * 3; // wrap same row right
            panels[focus_index]->TakeFocus();
            keypresses++;
            return true;
        }

        if (evt == Event::Character('j') || evt == Event::ArrowDown) {
            focus_index = (focus_index + 3) % 6; // jump row down
            panels[focus_index]->TakeFocus();
            keypresses++;
            return true;
        }

        if (evt == Event::Character('k') || evt == Event::ArrowUp) {
            focus_index = (focus_index + 3) % 6; // jump row up
            panels[focus_index]->TakeFocus();
            keypresses++;
            return true;
        }

        return false;
    });

    panels[focus_index]->TakeFocus();

    std::thread ticker([&] {
        while (running.load()) {
            // Drain telemetry
            while (auto p = sink.pop_for(std::chrono::milliseconds(0))) {
                state->update_from_packet(*p);
            }
            
            // Poll detector ledger
            auto new_anomalies = hook.detector().ledger();
            if (!new_anomalies.empty()) {
                std::lock_guard<std::recursive_mutex> lock(state->mutex);
                state->anomalies = new_anomalies;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(33));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(event_handler);
    running = false;
    
    // Stop interception if possible. The generate function may block until it's done.
    if (inference_thread.joinable()) inference_thread.join();
    if (ticker.joinable()) ticker.join();

    tui_log()->info("tui_hello exiting cleanly (keypresses={})", keypresses.load());
    return 0;
}
