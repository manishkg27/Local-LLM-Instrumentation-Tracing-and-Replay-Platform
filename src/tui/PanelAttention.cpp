#include "tui/PanelAttention.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace llm_tui {

ftxui::Component CreatePanelAttention(std::weak_ptr<AppState> state) {
    class Impl : public ftxui::ComponentBase {
        std::weak_ptr<AppState> state_;
        int pan_x_ = 4; // Center the 7x7 viewport in 16x16 grid
        int pan_y_ = 4;
        
    public:
        Impl(std::weak_ptr<AppState> state) : state_(state) {}
        
        ftxui::Element OnRender() override {
            using namespace ftxui;
            auto state = state_.lock();
            if (!state) {
                return text("No State") | dim;
            }
            std::lock_guard<std::recursive_mutex> lock(state->mutex);
            
            static constexpr std::string_view blocks[] = {
                " ", "░", "▒", "▓", "█"
            };
            
            // 1. Generate/populate a 16x16 attention matrix
            float full_matrix[16][16] = {0.0f};
            
            // Find relative time for dynamic perturbation
            uint64_t time_ns = 0;
            float attn_data[16] = {0.0f};
            bool has_patch = false;
            
            int target = state->target_layer_id;
            bool is_attention_target = false;
            
            if (target != -1) {
                for (const auto& node : state->visible_nodes) {
                    if (node->layer_id == target && node->type == LayerType::AttentionSelf) {
                        is_attention_target = true;
                        break;
                    }
                }
                if (!is_attention_target) {
                    return text("Select an Attention layer to view the matrix") | dim | center;
                }
            }
            
            int max_idx = state->is_replay_mode ? state->replay_cursor : static_cast<int>(state->packets.size()) - 1;
            
            if (max_idx >= 0 && max_idx < static_cast<int>(state->packets.size())) {
                time_ns = state->packets[max_idx].timestamp_ns;
                
                // Search for the latest TensorStats packet to extract real attn_patch
                for (int i = max_idx; i >= 0; --i) {
                    const auto& it = state->packets[i];
                    if (it.kind == PacketKind::TensorStats && it.layer_type == LayerType::AttentionSelf) {
                        if (target != -1 && it.layer_id != target) continue;
                        for (int j = 0; j < 16; ++j) {
                            attn_data[j] = it.attn_patch[j];
                        }
                        has_patch = true;
                        break;
                    }
                }
            }
            float t = static_cast<float>(time_ns % 1000000000) / 1e9f;
            
            // Populate the matrix
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    // Base: attention diagonal
                    float dist = std::abs(r - c);
                    float val = std::exp(-dist * 0.4f);
                    
                    // Wave perturbation
                    val += 0.15f * std::sin(r * 0.4f - c * 0.3f + t * 6.28f);
                    
                    // Blend in the real 4x4 patch in the middle if available
                    if (has_patch && r >= 6 && r < 10 && c >= 6 && c < 10) {
                        int pr = r - 6;
                        int pc = c - 6;
                        val = 0.5f * val + 0.5f * attn_data[pr * 4 + pc];
                    }
                    
                    full_matrix[r][c] = std::clamp(val, 0.0f, 1.0f);
                }
            }
            
            // 2. Extract the 7x7 slice based on pan coordinates
            Elements grid_rows;
            for (int r = 0; r < 7; ++r) {
                Elements row_cells;
                int map_r = std::clamp(pan_y_ + r, 0, 15);
                for (int c = 0; c < 7; ++c) {
                    int map_c = std::clamp(pan_x_ + c, 0, 15);
                    float raw_val = full_matrix[map_r][map_c];
                    float val = std::clamp(raw_val * state->attention_contrast, 0.0f, 1.0f);
                    
                    int block_idx = static_cast<int>(val * 4.0f);
                    block_idx = std::clamp(block_idx, 0, 4);
                    
                    // Color code cells
                    Color cell_color = Color::Blue;
                    if (val > 0.8f) {
                        cell_color = Color::Red;
                    } else if (val > 0.5f) {
                        cell_color = Color::Yellow;
                    } else if (val > 0.2f) {
                        cell_color = Color::Green;
                    }
                    
                    row_cells.push_back(text(std::string(blocks[block_idx])) | color(cell_color));
                    row_cells.push_back(text(" "));
                }
                grid_rows.push_back(hbox(std::move(row_cells)));
            }
            
            // Viewport coords text
            std::stringstream coord_ss;
            coord_ss << "Viewport: [" << pan_x_ << ":" << (pan_x_ + 6) 
                     << ", " << pan_y_ << ":" << (pan_y_ + 6) << "]";
            
            Elements info = {
                text("Contrast: ") | dim,
                text(std::to_string(state->attention_contrast).substr(0, 4)) | color(Color::Cyan),
                text("  ") | dim,
                text(coord_ss.str()) | color(Color::Green),
                text(state->attention_fullscreen ? " [FULLSCREEN]" : "  [F] Fullscreen") | dim
            };
            
            return vbox(Elements{
                text("   Attention Matrix Viewport (16x16)") | bold | center,
                separator(),
                vbox(std::move(grid_rows)) | center,
                separator(),
                hbox(std::move(info)) | center
            });
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
            
            if (event == ftxui::Event::Character('+') || event == ftxui::Event::Character('=')) {
                state->attention_contrast = std::min(5.0f, state->attention_contrast + 0.1f);
                return true;
            }
            if (event == ftxui::Event::Character('-') || event == ftxui::Event::Character('_')) {
                state->attention_contrast = std::max(0.1f, state->attention_contrast - 0.1f);
                return true;
            }
            
            // Toggle fullscreen
            if (event == ftxui::Event::Character('f') || event == ftxui::Event::Character('F')) {
                state->attention_fullscreen = !state->attention_fullscreen;
                return true;
            }
            
            // Viewport Panning (hjkl or arrows)
            if (event == ftxui::Event::Character('h') || event == ftxui::Event::ArrowLeft) {
                if (pan_x_ > 0) {
                    pan_x_--;
                    return true;
                }
            }
            if (event == ftxui::Event::Character('l') || event == ftxui::Event::ArrowRight) {
                if (pan_x_ < 9) {
                    pan_x_++;
                    return true;
                }
            }
            if (event == ftxui::Event::Character('k') || event == ftxui::Event::ArrowUp) {
                if (pan_y_ > 0) {
                    pan_y_--;
                    return true;
                }
            }
            if (event == ftxui::Event::Character('j') || event == ftxui::Event::ArrowDown) {
                if (pan_y_ < 9) {
                    pan_y_++;
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
