#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include "core/TelemetryPacket.hpp"
#include "core/AnomalyDetector.hpp"
#include "hook/LlamaInterceptor.hpp"

namespace llm_tui {

struct TreeNode : std::enable_shared_from_this<TreeNode> {
    std::string name;
    int layer_id = -1; // -1 if not a layer-specific sub-item
    LayerType type = LayerType::Unknown;
    bool expandable = false;
    bool expanded = false;
    int depth = 0;
    std::vector<std::shared_ptr<TreeNode>> children;
    std::weak_ptr<TreeNode> parent;
};

struct AppState {
    std::recursive_mutex mutex;
    
    std::string current_prompt;
    std::string generated_text;

    
    // Model metadata
    bool model_loaded = false;
    ModelTopology topology;
    
    // Tree navigation
    std::shared_ptr<TreeNode> tree_root;
    std::vector<std::shared_ptr<TreeNode>> visible_nodes;
    int selected_node_idx = 0;
    
    // Active execution tracking
    int active_layer_id = -1; // From incoming packets (current executing layer)
    
    // All packets processed so far
    std::vector<TelemetryPacket> packets;

    // Anomalies ledger entries copied from detector
    std::vector<LedgerEntry> anomalies;

    // TUI view options
    bool attention_fullscreen = false;
    float attention_contrast = 1.0f;
    std::string status_message = "Press [:] to type command, [?] or [:help] for commands";

    // Application Control
    bool capture_paused = false;
    int target_layer_id = -1; // Explicitly selected layer to filter other panels by
    int breakpoint_layer_id = -1; // Auto-pause if this layer is hit
    
    // Replay Mode Stepping
    bool is_replay_mode = false;
    int replay_cursor = -1; // -1 means show all (or inactive if !is_replay_mode)

    AppState();
    
    void update_from_packet(const TelemetryPacket& pkt);
    void build_tree();
    void update_visible_nodes();

    bool save_to_file(const std::string& filename);
    bool load_from_file(const std::string& filename);
};

} // namespace llm_tui
