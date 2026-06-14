#include "tui/PanelAttention.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace llm_tui {

// ---------------------------------------------------------------------------
// Helper: split a string by whitespace into word tokens.
// ---------------------------------------------------------------------------
static std::vector<std::string> split_words(std::string s) {
    std::string u2581 = "\xe2\x96\x81";
    size_t pos = 0;
    while ((pos = s.find(u2581, pos)) != std::string::npos) {
        s.replace(pos, u2581.length(), " ");
        pos += 1;
    }
    std::replace(s.begin(), s.end(), '\n', ' ');

    std::vector<std::string> words;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w) words.push_back(w);
    return words;
}

ftxui::Component CreatePanelAttention(std::weak_ptr<AppState> state) {
    class Impl : public ftxui::ComponentBase {
        std::weak_ptr<AppState> state_;
        int pan_x_ = 0;
        int pan_y_ = 0;

    public:
        Impl(std::weak_ptr<AppState> state) : state_(state) {}

        ftxui::Element OnRender() override {
            using namespace ftxui;
            auto state = state_.lock();
            if (!state) {
                return text("No State") | dim;
            }
            std::lock_guard<std::recursive_mutex> lock(state->mutex);

            // Block palette for the heatmap (from empty to solid)
            static constexpr std::string_view blocks[] = {
                "  ", "░░", "▒▒", "▓▓", "██"
            };

            // ------------------------------------------------------------------
            // 1. Gate: require an attention layer to be selected
            // ------------------------------------------------------------------
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

            // ------------------------------------------------------------------
            // 2. Build token labels from the current prompt + generated text
            // ------------------------------------------------------------------
            std::vector<std::string> tokens = split_words(state->current_prompt + state->generated_text);
            if (tokens.empty()) {
                tokens = {"T0", "T1", "T2", "T3"};
            }
            while (tokens.size() < 7) {
                tokens.insert(tokens.begin(), " ");
            }

            // ------------------------------------------------------------------
            // 3. Populate the 16×16 attention matrix
            // ------------------------------------------------------------------
            float full_matrix[16][16] = {0.0f};
            uint64_t time_ns = 0;
            float attn_data[49] = {0.0f};
            bool has_patch = false;

            int max_idx = state->is_replay_mode
                              ? state->replay_cursor
                              : static_cast<int>(state->packets.size()) - 1;

            if (max_idx >= 0 && max_idx < static_cast<int>(state->packets.size())) {
                time_ns = state->packets[max_idx].timestamp_ns;
                for (int i = max_idx; i >= 0; --i) {
                    const auto& it = state->packets[i];
                    if (it.kind == PacketKind::TensorStats &&
                        it.layer_type == LayerType::AttentionSelf) {
                        if (target != -1 && it.layer_id != target) continue;
                        
                        bool has_data = false;
                        for (int j = 0; j < 49; ++j) {
                            if (it.attn_patch[j] > 0.0f) {
                                has_data = true;
                                break;
                            }
                        }
                        
                        if (has_data) {
                            for (int j = 0; j < 49; ++j) {
                                attn_data[j] = it.attn_patch[j];
                            }
                            has_patch = true;
                            break;
                        }
                    }
                }
            }
            float t = static_cast<float>(time_ns % 1000000000) / 1e9f;

            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    float val = 0.0f;
                    if (has_patch && r < 7 && c < 7) {
                        val = attn_data[r * 7 + c];
                    }
                    full_matrix[r][c] = std::clamp(val, 0.0f, 1.0f);
                }
            }

            // ------------------------------------------------------------------
            // 4. Determine the viewport window
            // ------------------------------------------------------------------
            int kView = state->attention_fullscreen ? 7 : 4;
            int view_r0 = state->attention_fullscreen ? 0 : 3;
            int view_c0 = state->attention_fullscreen ? 0 : 3;

            // Collect visible token labels and compute uniform cell width
            std::vector<std::string> col_labels, row_labels;
            int start_idx = static_cast<int>(tokens.size()) - kView;
            for (int c = 0; c < kView; ++c)
                col_labels.push_back(tokens[start_idx + c]);
            for (int r = 0; r < kView; ++r)
                row_labels.push_back(tokens[start_idx + r]);

            // Cell width = max label width, clamped to [4, 10] for sanity
            int cell_w = 2;
            for (const auto& lbl : col_labels)
                cell_w = std::max(cell_w, static_cast<int>(lbl.size()));
            for (const auto& lbl : row_labels)
                cell_w = std::max(cell_w, static_cast<int>(lbl.size()));
            cell_w = std::clamp(cell_w + 1, 4, 10); // +1 padding

            // Width of the row-label gutter (left axis)
            int gutter_w = 0;
            for (const auto& lbl : row_labels)
                gutter_w = std::max(gutter_w, static_cast<int>(lbl.size()));
            gutter_w = std::clamp(gutter_w + 1, 4, 10);

            // Helper: right-pad a string to target width
            auto pad = [](const std::string& s, int w) -> std::string {
                if (static_cast<int>(s.size()) >= w) return s.substr(0, w);
                return s + std::string(w - s.size(), ' ');
            };
            // Helper: center a string within target width
            auto center_pad = [](const std::string& s, int w) -> std::string {
                if (static_cast<int>(s.size()) >= w) return s.substr(0, w);
                int total = w - static_cast<int>(s.size());
                int left = total / 2;
                int right = total - left;
                return std::string(left, ' ') + s + std::string(right, ' ');
            };

            // ------------------------------------------------------------------
            // 5. Build the grid — Column header row
            // ------------------------------------------------------------------
            Elements grid_rows;

            // "Tokens:" label row + column headers
            {
                Elements header_cells;
                // Gutter: label for the row axis
                header_cells.push_back(
                    text(pad("Tokens:", gutter_w)) | dim);
                // Column token labels
                for (int c = 0; c < kView; ++c) {
                    std::string lbl = "[" + col_labels[c] + "]";
                    header_cells.push_back(
                        text(center_pad(lbl, cell_w)) | bold | color(Color::Cyan));
                }
                grid_rows.push_back(hbox(std::move(header_cells)));
            }

            // ------------------------------------------------------------------
            // 6. Build each data row: row label + heatmap cells
            // ------------------------------------------------------------------
            for (int r = 0; r < kView; ++r) {
                Elements row_cells;
                // Row label (left axis)
                std::string rlbl = "[" + row_labels[r] + "]";
                row_cells.push_back(
                    text(pad(rlbl, gutter_w)) | bold | color(Color::Cyan));

                for (int c = 0; c < kView; ++c) {
                    float raw_val = full_matrix[view_r0 + r][view_c0 + c];
                    float val = std::clamp(raw_val * state->attention_contrast, 0.0f, 1.0f);

                    int block_idx = static_cast<int>(val * 4.0f);
                    block_idx = std::clamp(block_idx, 0, 4);

                    // Map value to a color ramp:
                    //   low → dim blue, mid-low → green, mid → yellow, high → red
                    Color cell_color = Color::Blue;
                    if (val > 0.8f) {
                        cell_color = Color::Red;
                    } else if (val > 0.5f) {
                        cell_color = Color::Yellow;
                    } else if (val > 0.2f) {
                        cell_color = Color::Green;
                    }

                    // Build the cell content: center the block chars within cell_w
                    std::string block_str(blocks[block_idx]);
                    std::string cell_content = center_pad(block_str, cell_w);
                    row_cells.push_back(text(cell_content) | color(cell_color));
                }
                grid_rows.push_back(hbox(std::move(row_cells)));
            }

            // ------------------------------------------------------------------
            // 7. Footer info bar
            // ------------------------------------------------------------------
            std::stringstream coord_ss;
            coord_ss << "Viewport Window: [" << view_c0 << "-" << (view_c0 + kView - 1)
                     << "] x [" << view_r0 << "-" << (view_r0 + kView - 1) << "]";

            Elements info = {
                text("Contrast: ") | dim,
                text(std::to_string(state->attention_contrast).substr(0, 4)) | color(Color::Cyan),
                text("  ") | dim,
                text(coord_ss.str()) | color(Color::Green),
                text(state->attention_fullscreen ? "  [FULLSCREEN]" : "  [F] Fullscreen") | dim,
            };

            Elements help = {
                text("[Focus + F]: Open Fullscreen") | dim,
                text("  ") | dim,
                text("[+/-]: Change Weight Contrast") | dim,
            };

            return vbox(Elements{
                text("  Attention Matrix Visualizer (HEAD 0)") | bold,
                separator(),
                vbox(std::move(grid_rows)),
                separator(),
                hbox(std::move(info)),
                hbox(std::move(help)),
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

            return false;
        }

        bool Focusable() const override { return true; }
    };

    return std::make_shared<Impl>(state);
}

} // namespace llm_tui
