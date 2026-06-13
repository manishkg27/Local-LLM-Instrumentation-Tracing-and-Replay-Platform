#include "tui/PanelTopology.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <algorithm>

namespace llm_tui {

ftxui::Component CreatePanelTopology(std::weak_ptr<AppState> state) {
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
            
            Elements tree_elements;
            
            if (state->visible_nodes.empty()) {
                return text("Empty Tree") | dim;
            }
            
            int total = static_cast<int>(state->visible_nodes.size());
            for (int i = 0; i < total; ++i) {
                const auto& node = state->visible_nodes[i];
                
                // 1. Indentation based on depth
                std::string indent = std::string(node->depth * 2, ' ');
                
                // 2. Expand/Collapse icon
                std::string expand_icon = "  ";
                if (node->expandable) {
                    expand_icon = node->expanded ? "󰅂 " : "󰅀 ";
                }
                
                // 3. Active execution marker ●
                bool is_active = (node->layer_id != -1 && node->layer_id == state->active_layer_id);
                Element active_marker = text(is_active ? "󰝤 " : "  ");
                if (is_active) {
                    active_marker = active_marker | color(Color::Green);
                }
                
                // 4. Node name / type label
                Element label = text(node->name);
                if (node->type != LayerType::Unknown) {
                    label = hbox(Elements{
                        label,
                        text(" (") | dim,
                        text(to_string(node->type)) | color(Color::Cyan),
                        text(")") | dim
                    });
                }
                
                // 5. Build full row
                Element row = hbox(Elements{
                    text(indent),
                    text(expand_icon) | color(Color::Yellow),
                    active_marker,
                    label
                });
                
                // 6. Focus/selected state
                bool is_selected = (i == state->selected_node_idx);
                if (is_selected) {
                    row = row | bold | color(Color::White) | bgcolor(Color::Blue);
                }
                
                tree_elements.push_back(row);
            }
            
            return vbox(std::move(tree_elements));
        }
        
        bool OnEvent(ftxui::Event event) override {
            auto state = state_.lock();
            if (!state) {
                return false;
            }
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            
            if (state->visible_nodes.empty()) {
                return false;
            }
            
            int total = static_cast<int>(state->visible_nodes.size());
            
            if (event == ftxui::Event::Character('j') || event == ftxui::Event::ArrowDown) {
                state->selected_node_idx = std::min(state->selected_node_idx + 1, total - 1);
                return true;
            }
            if (event == ftxui::Event::Character('k') || event == ftxui::Event::ArrowUp) {
                state->selected_node_idx = std::max(state->selected_node_idx - 1, 0);
                return true;
            }
            if (event == ftxui::Event::Character(' ') ) {
                auto& node = state->visible_nodes[state->selected_node_idx];
                state->target_layer_id = node->layer_id;
                return true;
            }
            if (event == ftxui::Event::Return) {
                auto& node = state->visible_nodes[state->selected_node_idx];
                if (node->expandable) {
                    node->expanded = !node->expanded;
                    state->update_visible_nodes();
                    return true;
                }
            }
            if (event == ftxui::Event::Character('h') || event == ftxui::Event::ArrowLeft) {
                auto& node = state->visible_nodes[state->selected_node_idx];
                if (node->expandable && node->expanded) {
                    node->expanded = false;
                    state->update_visible_nodes();
                    return true;
                } else {
                    auto parent = node->parent.lock();
                    if (parent && parent->depth >= 0) {
                        auto it = std::find(state->visible_nodes.begin(), state->visible_nodes.end(), parent);
                        if (it != state->visible_nodes.end()) {
                            state->selected_node_idx = static_cast<int>(std::distance(state->visible_nodes.begin(), it));
                            return true;
                        }
                    }
                }
            }
            if (event == ftxui::Event::Character('l') || event == ftxui::Event::ArrowRight) {
                auto& node = state->visible_nodes[state->selected_node_idx];
                if (node->expandable && !node->expanded) {
                    node->expanded = true;
                    state->update_visible_nodes();
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
