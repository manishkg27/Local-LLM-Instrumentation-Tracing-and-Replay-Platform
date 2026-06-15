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
        uint64_t frozen_seq_id_ = 0;
        
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
                text(" TIMESTAMP    ") | bold | color(Color::Yellow),
                text(" LAYER TYPE   ") | bold | color(Color::Yellow),
                text(" COMPUTE DEVICE   ") | bold | color(Color::Yellow),
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
            
            int base_max = total - 1;
            if (freeze_scroll_) {
                base_max = 0;
                for (int i = total - 1; i >= 0; --i) {
                    if (state->packets[i].sequence_id <= frozen_seq_id_) {
                        base_max = i;
                        break;
                    }
                }
            }
            
            // Ensure scroll offset is within bounds
            scroll_offset_ = std::clamp(scroll_offset_, 0, std::max(0, base_max));
            
            uint64_t start_ts = state->packets[0].timestamp_ns;
            int max_idx = base_max - scroll_offset_;
            
            int count = 0;
            for (int i = max_idx; i >= 0; --i) {
                if (count >= max_display) break;
                
                const auto& p = state->packets[i];
                count++;
                
                std::string id_str = std::to_string(p.sequence_id);
                if (id_str.size() < 5) id_str = std::string(5 - id_str.size(), ' ') + id_str;
                auto wall_now = std::chrono::system_clock::now();
                auto steady_now = std::chrono::steady_clock::now();
                auto steady_ns = std::chrono::nanoseconds(p.timestamp_ns);
                auto pkt_wall = wall_now - (steady_now.time_since_epoch() - steady_ns);
                
                auto time_t_val = std::chrono::system_clock::to_time_t(pkt_wall);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(pkt_wall.time_since_epoch()).count() % 1000;
                if (ms < 0) ms += 1000; // handle negative edge cases
                
                struct tm tm_val;
                localtime_r(&time_t_val, &tm_val);
                
                std::stringstream time_ss;
                time_ss << std::put_time(&tm_val, "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms;
                std::string time_str = time_ss.str() + "  ";
                
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
                std::string dev_str = (p.device == 1) ? "CUDA [GPU 0]" : "CPU (Fallback)";
                if (dev_str.size() < 17) dev_str += std::string(17 - dev_str.size(), ' ');
                
                // Color code packets: Warnings/Errors in red/yellow, normal in green
                Color row_color = Color::Green;
                if (p.kind == PacketKind::Anomaly) {
                    row_color = (p.severity == Severity::Warn) ? Color::Yellow : Color::Red;
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
                text("  [Space] Freeze  [j/k] Scroll  [F] Fullscreen") | dim
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
            
            // Fullscreen toggle removed, handled globally
            if (event == ftxui::Event::Character(' ')) {
                freeze_scroll_ = !freeze_scroll_;
                if (!freeze_scroll_) {
                    scroll_offset_ = 0;
                } else {
                    if (!state->packets.empty()) {
                        frozen_seq_id_ = state->packets.back().sequence_id;
                    }
                }
                return true;
            }
            
            if (freeze_scroll_) {
                if (event == ftxui::Event::Character('j') || event == ftxui::Event::ArrowDown) {
                    scroll_offset_ = std::max(0, scroll_offset_ - 1);
                    return true;
                }
                if (event == ftxui::Event::Character('k') || event == ftxui::Event::ArrowUp) {
                    scroll_offset_ += 1;
                    return true;
                }
            }
            
            return false;
        }

        bool Focusable() const override { return true; }
    };
    
    return std::make_shared<Impl>(state);
}

} // namespace llm_tui
