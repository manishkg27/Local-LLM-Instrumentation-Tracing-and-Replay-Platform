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
// Also handles the BOS token marker \xe2\x96\x81 (▁) used by many tokenizers.
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
        int pan_x_ = 0;  // horizontal offset for panning
        int pan_y_ = 0;  // vertical offset for panning

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
                tokens = {"[BOS]"};
            }

            // ------------------------------------------------------------------
            // 3. Find the latest attention data for the selected head/layer
            // ------------------------------------------------------------------
            // Priority: use the full attention matrix from the side-channel cache
            // when available (supports arbitrary viewport sizes and panning).
            // Fall back to the 7x7 attn_patch embedded in packets.

            uint64_t time_ns = 0;
            int captured_layer_id = -1;
            bool use_full_matrix = false;
            AttentionMatrix full_matrix;  // from side-channel cache

            // Try the full attention matrix cache first
            if (state->attn_cache_ptr && state->attn_cache_ptr->has_data()) {
                AttentionMatrix candidate = state->attn_cache_ptr->load();
                if (!candidate.empty()) {
                    // Filter by target layer if set
                    if (target == -1 || candidate.layer_id == target) {
                        full_matrix = candidate;
                        time_ns = full_matrix.timestamp_ns;
                        captured_layer_id = full_matrix.layer_id;
                        state->attention_seq_len = full_matrix.n_tok;
                        use_full_matrix = true;
                    }
                }
            }

            // Fall back to the 7x7 patch from packets
            float attn_data[49] = {0.0f};
            bool has_patch = false;

            if (!use_full_matrix) {
                int max_idx = state->is_replay_mode
                                  ? state->replay_cursor
                                  : static_cast<int>(state->packets.size()) - 1;

                // Search backwards for the most recent kq_soft_max attention packet
                if (max_idx >= 0 && max_idx < static_cast<int>(state->packets.size())) {
                    for (int i = max_idx; i >= 0; --i) {
                        const auto& it = state->packets[i];
                        if (it.kind == PacketKind::TensorStats &&
                            it.layer_type == LayerType::AttentionSelf) {
                            if (target != -1 && it.layer_id != target) continue;
                            if (it.head_idx >= 0 && it.head_idx != state->selected_head) continue;

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
                                time_ns = it.timestamp_ns;
                                captured_layer_id = it.layer_id;
                                if (it.attn_seq_len > 0) {
                                    state->attention_seq_len = it.attn_seq_len;
                                }
                                has_patch = true;
                                break;
                            }
                        }
                    }
                }
            }

            // ------------------------------------------------------------------
            // 4. Determine the visible window size
            // ------------------------------------------------------------------
            // When using the full matrix, the viewport can be up to the full matrix size.
            // When using the 7x7 patch, we show 4x4 (normal) or 7x7 (fullscreen).
            int matrix_rows = 0;
            int matrix_cols = 0;

            if (use_full_matrix && !full_matrix.empty()) {
                matrix_rows = full_matrix.n_tok;
                matrix_cols = full_matrix.n_kv;
            } else if (has_patch) {
                matrix_rows = 7;
                matrix_cols = 7;
            }

            int kView = 4;
            if (state->attention_fullscreen) {
                if (use_full_matrix && !full_matrix.empty()) {
                    // In fullscreen with full matrix, show up to 16x16 or the full matrix
                    kView = std::min(std::max(matrix_rows, matrix_cols), 16);
                } else {
                    kView = 7;
                }
            }

            // view_r0/view_c0: top-left offset into the matrix
            int view_r0 = 0;
            int view_c0 = 0;
            if (!use_full_matrix && has_patch && !state->attention_fullscreen) {
                // For the 7x7 patch fallback, show bottom-right 4x4
                view_r0 = 3;
                view_c0 = 3;
            }

            // ------------------------------------------------------------------
            // 5. Collect token labels for the visible window
            // ------------------------------------------------------------------
            // Use actual token count or fallback to generic labels
            int total_tokens = static_cast<int>(tokens.size());
            if (total_tokens < kView) {
                // Pad with empty tokens on the left
                while (static_cast<int>(tokens.size()) < kView) {
                    tokens.insert(tokens.begin(), "");
                }
                total_tokens = static_cast<int>(tokens.size());
            }

            // The column/row labels show the last kView tokens
            std::vector<std::string> col_labels, row_labels;
            int start_idx = total_tokens - kView + pan_x_;
            start_idx = std::clamp(start_idx, 0, std::max(0, total_tokens - kView));
            int start_row_idx = total_tokens - kView + pan_y_;
            start_row_idx = std::clamp(start_row_idx, 0, std::max(0, total_tokens - kView));

            for (int c = 0; c < kView && (start_idx + c) < static_cast<int>(tokens.size()); ++c)
                col_labels.push_back(tokens[start_idx + c]);
            for (int r = 0; r < kView && (start_row_idx + r) < static_cast<int>(tokens.size()); ++r)
                row_labels.push_back(tokens[start_row_idx + r]);
            // Pad if needed
            while (static_cast<int>(col_labels.size()) < kView)
                col_labels.push_back("T" + std::to_string(static_cast<int>(col_labels.size())));
            while (static_cast<int>(row_labels.size()) < kView)
                row_labels.push_back("T" + std::to_string(static_cast<int>(row_labels.size())));

            // Compute uniform cell width
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
            // 6. Build the grid — Column header row
            // ------------------------------------------------------------------
            Elements grid_rows;

            // "Tokens:" label row + column headers
            {
                Elements header_cells;
                header_cells.push_back(
                    text(pad("Tokens:", gutter_w)) | dim);
                for (int c = 0; c < kView; ++c) {
                    std::string lbl = "[" + col_labels[c] + "]";
                    header_cells.push_back(
                        text(center_pad(lbl, cell_w)) | bold | color(Color::Cyan));
                }
                grid_rows.push_back(hbox(std::move(header_cells)));
            }

            // ------------------------------------------------------------------
            // 7. Build each data row: row label + heatmap cells
            // ------------------------------------------------------------------
            for (int r = 0; r < kView; ++r) {
                Elements row_cells;
                std::string rlbl = "[" + row_labels[r] + "]";
                row_cells.push_back(
                    text(pad(rlbl, gutter_w)) | bold | color(Color::Cyan));

                for (int c = 0; c < kView; ++c) {
                    // Map viewport coordinates to the attention matrix
                    int mat_r = view_r0 + r;
                    int mat_c = view_c0 + c;
                    float raw_val = 0.0f;

                    if (use_full_matrix && !full_matrix.empty()) {
                        // Full matrix: direct indexing with bounds check
                        if (mat_r >= 0 && mat_r < full_matrix.n_tok &&
                            mat_c >= 0 && mat_c < full_matrix.n_kv) {
                            raw_val = full_matrix.data[mat_r * full_matrix.n_kv + mat_c];
                        }
                    } else if (has_patch) {
                        // 7x7 patch fallback
                        if (mat_r >= 0 && mat_r < 7 && mat_c >= 0 && mat_c < 7) {
                            raw_val = attn_data[mat_r * 7 + mat_c];
                        }
                    }
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

                    std::string block_str(blocks[block_idx]);
                    std::string cell_content = center_pad(block_str, cell_w);
                    row_cells.push_back(text(cell_content) | color(cell_color));
                }
                grid_rows.push_back(hbox(std::move(row_cells)));
            }

            // ------------------------------------------------------------------
            // 8. Footer info bar
            // ------------------------------------------------------------------
            std::stringstream coord_ss;
            coord_ss << "Viewport: [" << view_c0 << "-" << (view_c0 + kView - 1)
                     << "] x [" << view_r0 << "-" << (view_r0 + kView - 1) << "]";

            std::stringstream layer_ss;
            if (captured_layer_id >= 0) {
                layer_ss << "  Layer: " << captured_layer_id;
            }

            std::string head_label = "HEAD " + std::to_string(state->selected_head);
            std::string matrix_info;
            if (use_full_matrix && !full_matrix.empty()) {
                matrix_info = " [" + std::to_string(full_matrix.n_tok) + "x" +
                              std::to_string(full_matrix.n_kv) + " FULL]";
            }

            Elements info = {
                text("Contrast: ") | dim,
                text(std::to_string(state->attention_contrast).substr(0, 4)) | color(Color::Cyan),
                text("  ") | dim,
                text(coord_ss.str()) | color(Color::Green),
                text(layer_ss.str()) | color(Color::Yellow),
                text("  ") | dim,
                text(state->attention_fullscreen ? "[FULLSCREEN]" : "") | dim,
            };

            Elements help = {
                text("[F]: Fullscreen") | dim,
                text("  ") | dim,
                text("[+/-]: Contrast") | dim,
                text("  ") | dim,
                text("[H]: Cycle Head") | dim,
                text("  ") | dim,
                text("[Arrows]: Pan") | dim,
            };

            return vbox(Elements{
                text("  Attention Matrix Visualizer (" + head_label + ")" + matrix_info) | bold,
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
                pan_x_ = 0;
                pan_y_ = 0;
                return true;
            }

            // Cycle attention head
            if (event == ftxui::Event::Character('h') || event == ftxui::Event::Character('H')) {
                // Only intercept 'H' (shift-h) to avoid conflict with panel navigation 'h'
                if (event == ftxui::Event::Character('H')) {
                    state->selected_head = (state->selected_head + 1) % state->max_heads;
                    return true;
                }
                // lowercase 'h' is handled by parent for panel navigation
                return false;
            }

            // Pan with arrow keys
            if (event == ftxui::Event::ArrowLeft) {
                pan_x_ = std::max(pan_x_ - 1, -10);
                return true;
            }
            if (event == ftxui::Event::ArrowRight) {
                pan_x_ = std::min(pan_x_ + 1, 10);
                return true;
            }
            if (event == ftxui::Event::ArrowUp) {
                pan_y_ = std::max(pan_y_ - 1, -10);
                return true;
            }
            if (event == ftxui::Event::ArrowDown) {
                pan_y_ = std::min(pan_y_ + 1, 10);
                return true;
            }

            return false;
        }

        bool Focusable() const override { return true; }
    };

    return std::make_shared<Impl>(state);
}

} // namespace llm_tui