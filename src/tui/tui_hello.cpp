// =============================================================================
//  tui_hello.cpp
//  -----------------------------------------------------------------------------
//  Day 1 — Person 2 deliverable.
//
//  Renders a minimal but pretty FTXUI window with:
//    * a NerdFont-bordered panel (╔═╗║╚╝)
//    * a vim-style keybinding hint footer ([h/j/k/l] nav, [q] quit, [Tab] focus)
//    * a count of keypresses to prove the event loop is alive
//
//  The purpose is to confirm:
//    * FTXUI + spdlog + fmt link cleanly via vcpkg
//    * the terminal renders NerdFont glyphs (JetBrainsMono Nerd Font etc.)
//    * the keystroke loop runs at a sane frame rate
//
//  Run it and try:  j  j  k  l  Tab  h  q   (q to quit)
//
//  Compatibility note:
//  The vcpkg-installed FTXUI v5 only exposes `vbox(Elements)` (and same for
//  `hbox`), where `Elements` is a `std::vector<Element>`. The variadic
//  `vbox({...})` form used in many tutorials requires the optional
//  `take_any_args.hpp` shim, which is not exposed by default. We therefore
//  build the vector explicitly.
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

namespace {
std::shared_ptr<spdlog::logger> tui_log() {
    static auto lg = spdlog::stdout_color_mt("tui");
    lg->set_pattern("[%H:%M:%S.%e] [%^%l%$] [%n] %v");
    return lg;
}

// A simple "stat box" with a label, a number, and a NerdFont progress bar.
ftxui::Element stat_box(const std::string& label, std::string value,
                        int progress_0_to_10) {
    using namespace ftxui;
    static constexpr std::string_view bars[11] = {
        "▱▱▱▱▱▱▱▱▱▱",  // 0
        "▰▱▱▱▱▱▱▱▱▱",  // 1
        "▰▰▱▱▱▱▱▱▱▱",
        "▰▰▰▱▱▱▱▱▱▱",
        "▰▰▰▰▱▱▱▱▱▱",
        "▰▰▰▰▰▱▱▱▱▱",
        "▰▰▰▰▰▰▱▱▱▱",
        "▰▰▰▰▰▰▰▱▱▱",
        "▰▰▰▰▰▰▰▰▱▱",
        "▰▰▰▰▰▰▰▰▰▱",
        "▰▰▰▰▰▰▰▰▰▰",  // 10
    };
    int p = std::clamp(progress_0_to_10, 0, 10);
    ftxui::Elements rows = {
        text(label) | bold,
        text(value) | color(Color::Cyan),
        text(std::string(bars[p])),
    };
    return vbox(std::move(rows)) | border | size(WIDTH, EQUAL, 24);
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
    tui_log()->info("tui_hello (Day 1) starting");

    auto screen = ScreenInteractive::TerminalOutput();

    // Shared state
    std::atomic<int> keypresses{0};
    std::atomic<int> progress{3};      // 0..10
    std::atomic<int> focus_idx{0};     // 0..4 (which panel is "active")
    std::atomic<bool> running{true};

    // The "renderer" lambda re-evaluates the DOM on every event.
    auto renderer = Renderer([&] {
        const int kp = keypresses.load();
        const int pr = progress.load();
        const int fi = focus_idx.load();

        // 5 mock panels arranged like the final TUI mockup.
        ftxui::Elements panel_row1 = {
            stat_box("1. MODEL TOPOLOGY",
                     fi == 0 ? "▶ Active" : "idle", 6) |
                (fi == 0 ? borderHeavy : border),
            stat_box("2. LIVE PACKET STREAM",
                     fi == 1 ? "▶ Active" : "idle", 4) |
                (fi == 1 ? borderHeavy : border),
            stat_box("3. ATTENTION MATRIX",
                     fi == 2 ? "▶ Active" : "idle", 7) |
                (fi == 2 ? borderHeavy : border),
        };
        ftxui::Elements panel_row2 = {
            stat_box("4. RUNTIME METRICS",
                     fi == 3 ? "▶ Active" : "idle", pr) |
                (fi == 3 ? borderHeavy : border),
            stat_box("5. ANOMALY LEDGER",
                     fi == 4 ? "▶ Active" : "idle", 2) |
                (fi == 4 ? borderHeavy : border),
            stat_box("STATUS", "Day 1 OK", 10) | border,
        };

        ftxui::Elements rows = {
            header(),
            separator(),
            hbox(std::move(panel_row1)),
            hbox(std::move(panel_row2)),
            separator(),
            footer(kp),
        };
        return vbox(std::move(rows));
    });

    // Event handler — vim-style
    auto component = CatchEvent(renderer, [&](Event evt) {
        if (evt == Event::Character('q') || evt == Event::Escape) {
            running = false;
            screen.ExitLoopClosure()();
            return true;
        }
        if (evt == Event::Tab) {
            focus_idx = (focus_idx.load() + 1) % 5;
            keypresses++;
            return true;
        }
        if (evt == Event::Character('h') || evt == Event::ArrowLeft) {
            int f = focus_idx.load();
            focus_idx = (f + 4) % 5;       // wrap backwards
            keypresses++;
            return true;
        }
        if (evt == Event::Character('l') || evt == Event::ArrowRight) {
            int f = focus_idx.load();
            focus_idx = (f + 1) % 5;
            keypresses++;
            return true;
        }
        if (evt == Event::Character('j') || evt == Event::ArrowDown) {
            int f = focus_idx.load();
            focus_idx = std::min(4, f + 1);
            keypresses++;
            return true;
        }
        if (evt == Event::Character('k') || evt == Event::ArrowUp) {
            int f = focus_idx.load();
            focus_idx = std::max(0, f - 1);
            keypresses++;
            return true;
        }
        if (evt == Event::Character('+') || evt == Event::Character('=')) {
            int p = progress.load();
            progress = std::min(10, p + 1);
            keypresses++;
            return true;
        }
        if (evt == Event::Character('-') || evt == Event::Character('_')) {
            int p = progress.load();
            progress = std::max(0, p - 1);
            keypresses++;
            return true;
        }
        return false;
    });

    // Tick the screen at ~30 FPS even with no input, so the focus highlight
    // redraws when state changes elsewhere.
    std::thread ticker([&] {
        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(component);
    running = false;
    if (ticker.joinable()) ticker.join();

    tui_log()->info("tui_hello exiting cleanly (keypresses={})", keypresses.load());
    return 0;
}
