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
//  Keybinding model
//  -----------------
//  The TUI renders a 3+3 grid:
//
//      row 0:  [1. MODEL TOPOLOGY] [2. LIVE PACKET STREAM] [3. ATTENTION MATRIX]
//      row 1:  [4. RUNTIME METRICS] [5. ANOMALY LEDGER]   [6. STATUS]
//
//  so the focus cursor is 2-D: (row, col).
//    h / ←  move left  within the SAME row (col -1, wrap 2→0)
//    l / →  move right within the SAME row (col +1, wrap 0→2)
//    j / ↓  jump to the OTHER row, same column (row toggles 0↔1)
//    k / ↑  jump to the OTHER row, same column (same as j here — only 2 rows)
//    Tab    cycle through all 6 in linear order (1→2→3→4→5→6→1…)
//    q/Esc  quit
//    + / -  grow / shrink the "4. RUNTIME METRICS" progress bar
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

    // ---- Shared state -------------------------------------------------
    std::atomic<int> keypresses{0};
    std::atomic<int> progress{3};      // 0..10 (drives the 4. RUNTIME METRICS bar)
    // Focus is 2-D because the TUI is a 3+3 grid.
    //   row ∈ {0, 1},  col ∈ {0, 1, 2}
    //   linear index fi = row*3 + col  ∈ {0, 1, 2, 3, 4, 5}
    std::atomic<int> focus_row{0};
    std::atomic<int> focus_col{0};
    std::atomic<bool> running{true};

    // ---- Renderer -----------------------------------------------------
    auto renderer = Renderer([&] {
        const int kp  = keypresses.load();
        const int pr  = progress.load();
        const int fi  = focus_row.load() * 3 + focus_col.load();

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

    // ---- Event handler — 2-D navigation ------------------------------
    auto component = CatchEvent(renderer, [&](Event evt) {
        // q / Esc  →  quit
        if (evt == Event::Character('q') || evt == Event::Escape) {
            running = false;
            screen.ExitLoopClosure()();
            return true;
        }

        // Tab  →  cycle all 6 panels in linear order 0→1→2→3→4→5→0…
        if (evt == Event::Tab) {
            int linear = focus_row.load() * 3 + focus_col.load();
            linear = (linear + 1) % 6;
            focus_row = linear / 3;
            focus_col = linear % 3;
            keypresses++;
            return true;
        }

        // h / ←  →  move left  in the SAME row (col -1, wrap 2→0)
        if (evt == Event::Character('h') || evt == Event::ArrowLeft) {
            int c = focus_col.load();
            focus_col = (c + 2) % 3;   // modular -1 ≡ +2 (mod 3)
            keypresses++;
            return true;
        }

        // l / →  →  move right in the SAME row (col +1, wrap 0→2)
        if (evt == Event::Character('l') || evt == Event::ArrowRight) {
            int c = focus_col.load();
            focus_col = (c + 1) % 3;
            keypresses++;
            return true;
        }

        // j / ↓  →  jump to the OTHER row, same column  (row toggles 0↔1)
        if (evt == Event::Character('j') || evt == Event::ArrowDown) {
            focus_row = 1 - focus_row.load();   // 0→1, 1→0
            keypresses++;
            return true;
        }

        // k / ↑  →  jump to the OTHER row, same column  (same as j for 2 rows)
        if (evt == Event::Character('k') || evt == Event::ArrowUp) {
            focus_row = 1 - focus_row.load();
            keypresses++;
            return true;
        }

        // + / =  →  grow progress bar (0..10)
        if (evt == Event::Character('+') || evt == Event::Character('=')) {
            int p = progress.load();
            progress = std::min(10, p + 1);
            keypresses++;
            return true;
        }

        // - / _  →  shrink progress bar (0..10)
        if (evt == Event::Character('-') || evt == Event::Character('_')) {
            int p = progress.load();
            progress = std::max(0, p - 1);
            keypresses++;
            return true;
        }

        return false;   // not handled, let FTXUI's default handling see it
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
