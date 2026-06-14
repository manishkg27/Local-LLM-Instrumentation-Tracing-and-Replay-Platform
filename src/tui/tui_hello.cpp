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

// A simple "stat box" for STATUS
ftxui::Element stat_box(const std::string& label, std::string value) {
    using namespace ftxui;
    ftxui::Elements rows = {
        text(label) | bold,
        text(value) | color(Color::Cyan),
    };
    return vbox(std::move(rows)) | center;
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
    llm_tui::RingBuffer<llm_tui::TelemetryPacket> sink(1024);
    llm_tui::LlamaInterceptor hook;
    hook.set_sink(&sink);

    std::thread inference_thread([&]() {
        state->current_prompt = "Hello, what is the meaning of life?";
        std::string model_path = "/home/manish/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf";
        if (const char* env = std::getenv("LLM_TUI_MODEL"); env && *env) {
            model_path = env;
        }

        if (hook.load(model_path, 2048, 0)) {
            // continuously generate to keep TUI alive
            while (running.load()) {
                hook.generate(state->current_prompt, [&](const std::string& /*token*/) {});
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
        Renderer([] { return stat_box("STATUS", "Day 1 OK"); })
    };

    // Decorate panels with borders and highlights
    std::vector<ftxui::Component> decorated_panels;
    std::vector<std::string> titles = {
        "1. MODEL TOPOLOGY", "2. LIVE PACKET STREAM", "3. ATTENTION MATRIX",
        "4. RUNTIME METRICS", "5. ANOMALY LEDGER", "6. STATUS"
    };

    for (size_t i = 0; i < panels.size(); ++i) {
        auto wrapped = Renderer(panels[i], [i, &focus_index, &panels, &titles] {
            bool focused = (static_cast<int>(i) == focus_index);
            return window(text(titles[i]) | (focused ? bold : dim), panels[i]->Render())
                   | size(WIDTH, EQUAL, 40)
                   | size(HEIGHT, EQUAL, 24)
                   | (focused ? borderHeavy : border);
        });
        decorated_panels.push_back(wrapped);
    }

    // Layout
    auto main_container = Container::Vertical({
        Container::Horizontal({decorated_panels[0], decorated_panels[1], decorated_panels[2]}),
        Container::Horizontal({decorated_panels[3], decorated_panels[4], decorated_panels[5]})
    });

    auto root = Renderer(main_container, [&] {
        return vbox({
            header(),
            separator(),
            main_container->Render(),
            separator(),
            footer(keypresses.load()),
        });
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
