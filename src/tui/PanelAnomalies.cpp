#include "tui/PanelAnomalies.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace llm_tui {

ftxui::Component CreatePanelAnomalies(std::weak_ptr<AppState> state) {
    class Impl : public ftxui::ComponentBase {
        std::weak_ptr<AppState> state_;
        int scroll_offset_ = 0;
        
    public:
        Impl(std::weak_ptr<AppState> state) : state_(state) {}
        
        ftxui::Element OnRender() override {
            using namespace ftxui;
            auto state = state_.lock();
            if (!state) {
                return text("No State") | dim;
            }
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            
            Elements rows;
            
            auto get_severity_el = [](Severity sev) {
                switch (sev) {
                    case Severity::Info:  return text(" ℹ ") | color(Color::Blue);
                    case Severity::Warn:  return text(" ⚠ ") | color(Color::Yellow) | bold;
                    case Severity::Error: return text(" ✖ ") | color(Color::Red) | bold;
                }
                return text(" ? ");
            };
            
            if (state->packets.empty()) {
                rows.push_back(text("No anomalies detected yet") | dim);
                return vbox(std::move(rows));
            }
            
            int total = static_cast<int>(state->anomalies.size());
            if (total == 0) {
                rows.push_back(text("✓ No numerical anomalies detected") | color(Color::Green));
                return vbox(std::move(rows));
            }
            
            // Clamp scroll offset
            scroll_offset_ = std::clamp(scroll_offset_, 0, std::max(0, total - 10));
            
            uint64_t start_ts = state->packets.empty() ? 0 : state->packets[0].timestamp_ns;
            
            // Render visible anomalies (newest at top, so reverse order)
            int count = 0;
            for (int i = total - 1 - scroll_offset_; i >= 0 && count < 10; --i, ++count) {
                const auto& entry = state->anomalies[i];
                
                Element icon = get_severity_el(entry.severity);
                
                // Truncate message if too long for display
                std::string msg = entry.message;
                if (msg.size() > 42) msg = msg.substr(0, 39) + "...";
                
                double rel_time = 0.0;
                if (start_ts > 0 && entry.timestamp_ns > start_ts) {
                    rel_time = (entry.timestamp_ns - start_ts) / 1e9;
                }
                std::stringstream time_ss;
                time_ss << std::fixed << std::setprecision(2) << rel_time << "s";
                
                rows.push_back(hbox(Elements{
                    text("[" + time_ss.str() + "]") | dim,
                    icon,
                    text(msg)
                }));
            }
            
            Elements footer_info = {
                text("Total: ") | dim,
                text(std::to_string(total)) | color(Color::RedLight),
                text("  [j/k] scroll  [g/G] top/bottom") | dim
            };
            
            return vbox(Elements{
                vbox(std::move(rows)),
                filler(),
                separator(),
                hbox(std::move(footer_info)) | center
            });
        }
        
        bool OnEvent(ftxui::Event event) override {
            auto state = state_.lock();
            if (!state) {
                return false;
            }
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            int total = static_cast<int>(state->anomalies.size());
            if (total == 0) return false;
            
            if (event == ftxui::Event::Character('j') || event == ftxui::Event::ArrowDown) {
                scroll_offset_ = std::min(scroll_offset_ + 1, std::max(0, total - 1));
                return true;
            }
            if (event == ftxui::Event::Character('k') || event == ftxui::Event::ArrowUp) {
                scroll_offset_ = std::max(scroll_offset_ - 1, 0);
                return true;
            }
            if (event == ftxui::Event::Character('g')) {
                scroll_offset_ = 0;
                return true;
            }
            if (event == ftxui::Event::Character('G')) {
                scroll_offset_ = std::max(0, total - 1);
                return true;
            }
            return false;
        }
    };
    
    return std::make_shared<Impl>(state);
}

} // namespace llm_tui
