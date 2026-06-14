#include "tui/PanelMetrics.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace llm_tui {

ftxui::Component CreatePanelMetrics(std::weak_ptr<AppState> state) {
    class Impl : public ftxui::ComponentBase {
        std::weak_ptr<AppState> state_;
        
    public:
        Impl(std::weak_ptr<AppState> state) : state_(state) {}
        
        ftxui::Element OnRender() override {
            using namespace ftxui;
            auto state = state_.lock();
            if (!state) {
                return text("No State") | dim;
            }
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            
            int target_layer = state->target_layer_id;
            if (target_layer == -1) {
                target_layer = state->active_layer_id;
            }
            if (target_layer == -1) {
                return text("Waiting for execution or press [Space] to select") | dim | center;
            }
            
            std::string layer_name = "Layer " + std::to_string(target_layer);
            LayerType layer_type = (state->target_layer_id != -1) ? state->target_layer_type : LayerType::Unknown;
            // Best-effort to find the name/type
            for (const auto& node : state->visible_nodes) {
                if (node->layer_id == target_layer && (layer_type == LayerType::Unknown || node->type == layer_type)) {
                    layer_name = node->name;
                    if (layer_type == LayerType::Unknown && node->type != LayerType::Unknown) {
                        layer_type = node->type;
                    }
                }
            }
            
            // Search back in history for stats & latency packets of this layer
            float latency_us = 0.0f;
            float sparsity = 0.0f;
            float mean = 0.0f;
            float max_abs = 0.0f;
            Shape shape = {0, 0, 0, 0};
            bool has_latency = false;
            bool has_stats = false;
            bool has_topology = false;
            
            int max_idx = state->is_replay_mode ? state->replay_cursor : static_cast<int>(state->packets.size()) - 1;
            
            if (max_idx >= 0 && max_idx < static_cast<int>(state->packets.size())) {
                for (int i = max_idx; i >= 0; --i) {
                    const auto& it = state->packets[i];
                    if (it.layer_id == target_layer) {
                        // If we specifically requested a layer type (e.g., MLP), enforce it
                        bool type_matches = (state->target_layer_id == -1 || state->target_layer_type == LayerType::Unknown || it.layer_type == state->target_layer_type);
                        
                        if (type_matches) {
                            if (it.kind == PacketKind::LayerLatency && !has_latency) {
                                latency_us = it.latency_us;
                                has_latency = true;
                            } else if (it.kind == PacketKind::TensorStats && !has_stats) {
                                sparsity = it.sparsity;
                                mean = it.mean;
                                max_abs = it.max_abs;
                                shape = it.shape;
                                has_stats = true;
                            }
                        }
                        if (it.kind == PacketKind::Topology && !has_topology) {
                            shape = it.shape;
                            has_topology = true;
                        }
                    }
                    if (has_latency && has_stats && has_topology) break;
                }
            }
            
            // Build the layout
            Elements details;
            details.push_back(hbox(Elements{
                text("Layer: ") | dim,
                text(layer_name) | bold | color(Color::Green)
            }));
            
            std::string type_str = (layer_type != LayerType::Unknown) ? to_string(layer_type) : "Container";
            details.push_back(hbox(Elements{
                text("Type:  ") | dim,
                text(type_str) | color(Color::Cyan)
            }));
            
            if (target_layer != -1) {
                // Check if this layer has any recorded anomalies
                Severity highest_sev = Severity::Info;
                bool has_anomaly = false;
                for (const auto& entry : state->anomalies) {
                    if (entry.layer_id == target_layer) {
                        has_anomaly = true;
                        if (entry.severity > highest_sev) {
                            highest_sev = entry.severity;
                        }
                    }
                }
                
                Element status_el;
                if (!has_anomaly) {
                    status_el = text("✓ Normal") | color(Color::Green) | bold;
                } else if (highest_sev == Severity::Error) {
                    status_el = text("✖ Critical") | color(Color::Red) | bold;
                } else {
                    status_el = text("⚠ Warning") | color(Color::Yellow) | bold;
                }
                
                details.push_back(hbox(Elements{
                    text("Status:") | dim,
                    status_el
                }));
                
                details.push_back(hbox(Elements{
                    text("ID:    ") | dim,
                    text(std::to_string(target_layer))
                }));
                
                // Shape info
                std::stringstream shape_ss;
                shape_ss << "[" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "]";
                details.push_back(hbox(Elements{
                    text("Shape: ") | dim,
                    text(shape_ss.str())
                }));
                
                // Latency
                std::stringstream lat_ss;
                lat_ss << std::fixed << std::setprecision(1) << latency_us << " us";
                details.push_back(hbox(Elements{
                    text("Time:  ") | dim,
                    text(has_latency ? lat_ss.str() : "0.0 us (idle)") | color(has_latency ? Color::White : Color::GrayDark)
                }));
                
                // Sparsity bar
                int filled = static_cast<int>(sparsity * 10.0f);
                std::string bar = "";
                for (int i = 0; i < 10; ++i) {
                    bar += (i < filled) ? "🟩" : "⬜";
                }
                std::stringstream sp_ss;
                sp_ss << std::fixed << std::setprecision(1) << (sparsity * 100.0f) << "%";
                
                details.push_back(hbox(Elements{
                    text("Sparse:") | dim,
                    text(bar + " "),
                    text(sp_ss.str()) | color(Color::Yellow)
                }));
                
                // Mean/Max
                std::stringstream stats_ss;
                stats_ss << "μ=" << std::fixed << std::setprecision(4) << mean
                         << "  |x|_max=" << max_abs;
                details.push_back(hbox(Elements{
                    text("Stats: ") | dim,
                    text(has_stats ? stats_ss.str() : "N/A") | color(Color::GrayLight)
                }));
            } else {
                // Global model metadata
                details.push_back(separator());
                details.push_back(text("Global Model Metrics") | bold | color(Color::Yellow));
                details.push_back(hbox(Elements{
                    text("Layers: ") | dim,
                    text(std::to_string(state->topology.n_layer))
                }));
                details.push_back(hbox(Elements{
                    text("Params: ") | dim,
                    text(std::to_string(state->topology.n_params / 1e9).substr(0, 4) + " B")
                }));
            }
            
            return vbox(std::move(details));
        }
        
        bool OnEvent(ftxui::Event event) override {
            return false;
        }

        bool Focusable() const override { return true; }
    };
    
    return std::make_shared<Impl>(state);
}

} // namespace llm_tui
