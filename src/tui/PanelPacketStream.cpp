#include "tui/PanelPacketStream.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

namespace llm_tui {

ftxui::Component CreatePanelPacketStream(std::weak_ptr<AppState> state) {
    class Impl : public ftxui::ComponentBase {
        std::weak_ptr<AppState> state_;
        bool freeze_scroll_ = false;
        int scroll_offset_ = 0; // scroll offset from the bottom (0 = show latest)
        
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
            
            // Header columns: ID | TIME (s) | TYPE | DEVICE
            std::string freeze_indicator = freeze_scroll_ ? " [FROZEN] " : " [LIVE] ";
            rows.push_back(hbox(Elements{
                text(" ID    ") | bold | color(Color::Yellow),
                text(" TIME (s) ") | bold | color(Color::Yellow),
                text(" TYPE                ") | bold | color(Color::Yellow),
                text(" DEVICE ") | bold | color(Color::Yellow),
                filler(),
                text(freeze_indicator) | bold | color(freeze_scroll_ ? Color::Red : Color::Green)
            }));
            rows.push_back(separator());
            
            int total = static_cast<int>(state->packets.size());
            if (total == 0) {
                rows.push_back(text("No packets received yet") | dim);
                return vbox(std::move(rows));
            }
            
            // Limit visible area to about 9-10 lines depending on panel height
            // We can determine the visible packets dynamically based on scroll_offset_
            const int max_display = 10;
            
            // If not frozen, auto-scroll to the bottom (offset = 0)
            if (!freeze_scroll_) {
                scroll_offset_ = 0;
            }
            
            // Ensure scroll offset is within bounds
            scroll_offset_ = std::clamp(scroll_offset_, 0, std::max(0, total - max_display));
            
            int end_idx = total - scroll_offset_;
            int start_idx = std::max(0, end_idx - max_display);
            
            uint64_t start_ts = state->packets[0].timestamp_ns;
            
            for (int i = start_idx; i < end_idx; ++i) {
                const auto& p = state->packets[i];
                
                std::string id_str = std::to_string(p.sequence_id);
                if (id_str.size() < 5) id_str = std::string(5 - id_str.size(), ' ') + id_str;
                
                double rel_time = (p.timestamp_ns - start_ts) / 1e9;
                std::stringstream time_ss;
                time_ss << std::fixed << std::setprecision(3) << rel_time;
                std::string time_str = time_ss.str();
                if (time_str.size() < 9) time_str = std::string(9 - time_str.size(), ' ') + time_str;
                
                // Construct type description (combines PacketKind and LayerType if relevant)
                std::string type_str;
                if (p.kind == PacketKind::TensorStats) {
                    type_str = std::string(to_string(p.kind)) + " (" + to_string(p.layer_type) + ")";
                } else if (p.kind == PacketKind::LayerLatency && p.layer_id != -1) {
                    type_str = std::string(to_string(p.kind)) + " (" + to_string(p.layer_type) + ")";
                } else {
                    type_str = to_string(p.kind);
                }
                if (type_str.size() < 20) type_str += std::string(20 - type_str.size(), ' ');
                
                std::string dev_str = (p.device == 1) ? "CUDA" : "CPU";
                if (dev_str.size() < 7) dev_str = std::string(7 - dev_str.size(), ' ') + dev_str;
                
                // Color code packets: Warnings/Errors in red/yellow, normal in cyan/green
                Color row_color = Color::White;
                if (p.kind == PacketKind::Anomaly) {
                    row_color = Color::Red;
                } else if (p.kind == PacketKind::TokenStart) {
                    row_color = Color::GreenLight;
                } else if (p.kind == PacketKind::TokenEnd) {
                    row_color = Color::YellowLight;
                } else if (p.kind == PacketKind::TensorStats) {
                    row_color = Color::CyanLight;
                }
                
                rows.push_back(hbox(Elements{
                    text(id_str + " "),
                    text(time_str + " "),
                    text(type_str) | color(row_color),
                    text(dev_str)
                }));
            }
            
            // Legend at the bottom
            rows.push_back(filler());
            rows.push_back(separator());
            rows.push_back(hbox(Elements{
                text("Packets: ") | dim,
                text(std::to_string(total)) | color(Color::Cyan),
                text("  [Space] Freeze  [j/k or Up/Down] Scroll (when frozen)") | dim
            }) | center);
            
            return vbox(std::move(rows));
        }
        
        bool OnEvent(ftxui::Event event) override {
            if (!Focused()) {
                return false;
            }
            
            auto state = state_.lock();
            if (!state) {
                return false;
            }
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            int total = static_cast<int>(state->packets.size());
            
            if (event == ftxui::Event::Character(' ')) {
                freeze_scroll_ = !freeze_scroll_;
                if (!freeze_scroll_) {
                    scroll_offset_ = 0;
                }
                return true;
            }
            
            if (freeze_scroll_) {
                if (event == ftxui::Event::Character('j') || event == ftxui::Event::ArrowDown) {
                    scroll_offset_ = std::max(0, scroll_offset_ - 1);
                    return true;
                }
                if (event == ftxui::Event::Character('k') || event == ftxui::Event::ArrowUp) {
                    scroll_offset_ = std::min(total - 10, scroll_offset_ + 1);
                    return true;
                }
            }
            
            return false;
        }
    };
    
    return std::make_shared<Impl>(state);
}

} // namespace llm_tui
