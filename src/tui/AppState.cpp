#include "tui/AppState.hpp"
#include <algorithm>

namespace llm_tui {

AppState::AppState() {
    tree_root = std::make_shared<TreeNode>();
    tree_root->name = "Root";
    tree_root->depth = -1;
    build_tree();
}

void AppState::update_from_packet(const TelemetryPacket& pkt) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    
    // Store packet
    packets.push_back(pkt);

    if (pkt.kind == PacketKind::Topology) {
        if (pkt.layer_id == -1) {
            model_loaded = true;
            topology.n_layer = pkt.shape[0];
            topology.n_head = pkt.shape[1];
            topology.n_embd = pkt.shape[2];
            topology.n_vocab = pkt.shape[3];
            topology.n_params = static_cast<std::uint64_t>(pkt.sigma * 1e9);
            topology.name = "Qwen2.5-Coder-3B"; // Set default or check if we can get it
            build_tree();
        } else {
            // Update individual layer types from incoming topology packets
            if (tree_root && !tree_root->children.empty()) {
                // Find layers node (usually second child of root)
                std::shared_ptr<TreeNode> layers_node = nullptr;
                for (auto& child : tree_root->children) {
                    if (child->name == "layers") {
                        layers_node = child;
                        break;
                    }
                }
                if (layers_node && pkt.layer_id < static_cast<int>(layers_node->children.size())) {
                    auto& layer_item = layers_node->children[pkt.layer_id];
                    layer_item->type = pkt.layer_type;
                    
                    // Rebuild sub-items based on actual layer type
                    layer_item->children.clear();
                    
                    auto attn = std::make_shared<TreeNode>();
                    attn->name = to_string(pkt.layer_type);
                    attn->layer_id = pkt.layer_id;
                    attn->type = pkt.layer_type;
                    attn->depth = 2;
                    attn->parent = layer_item;
                    layer_item->children.push_back(attn);

                    auto mlp = std::make_shared<TreeNode>();
                    mlp->name = "MLP (SwiGLU)";
                    mlp->layer_id = pkt.layer_id;
                    mlp->type = LayerType::Mlp;
                    mlp->depth = 2;
                    mlp->parent = layer_item;
                    layer_item->children.push_back(mlp);

                    auto norm = std::make_shared<TreeNode>();
                    norm->name = "RMSNorm";
                    norm->layer_id = pkt.layer_id;
                    norm->type = LayerType::RmsNorm;
                    norm->depth = 2;
                    norm->parent = layer_item;
                    layer_item->children.push_back(norm);
                }
            }
        }
    } else if (pkt.kind == PacketKind::LayerLatency || pkt.kind == PacketKind::TensorStats) {
        if (pkt.layer_id >= 0) {
            active_layer_id = pkt.layer_id;
        }
    }
}

void AppState::build_tree() {
    tree_root->children.clear();
    visible_nodes.clear();
    selected_node_idx = 0;

    if (!model_loaded) {
        auto placeholder = std::make_shared<TreeNode>();
        placeholder->name = "Waiting for model load...";
        placeholder->depth = 0;
        placeholder->parent = tree_root;
        tree_root->children.push_back(placeholder);
        visible_nodes.push_back(placeholder);
        return;
    }

    // 1. embed_tokens
    auto embed = std::make_shared<TreeNode>();
    embed->name = "embed_tokens";
    embed->type = LayerType::Embedding;
    embed->depth = 0;
    embed->parent = tree_root;
    tree_root->children.push_back(embed);

    // 2. layers
    auto layers = std::make_shared<TreeNode>();
    layers->name = "layers";
    layers->depth = 0;
    layers->expandable = true;
    layers->expanded = true; // start expanded by default for layers container
    layers->parent = tree_root;
    
    for (int i = 0; i < topology.n_layer; ++i) {
        auto layer_item = std::make_shared<TreeNode>();
        layer_item->name = "layers." + std::to_string(i);
        layer_item->layer_id = i;
        layer_item->depth = 1;
        layer_item->expandable = true;
        layer_item->expanded = false; // keep individual layers collapsed initially
        layer_item->parent = layers;
        
        // Add default sub-items (will be updated dynamically on load)
        auto attn = std::make_shared<TreeNode>();
        attn->name = "Attn (Self)";
        attn->layer_id = i;
        attn->type = LayerType::AttentionSelf;
        attn->depth = 2;
        attn->parent = layer_item;
        layer_item->children.push_back(attn);

        auto mlp = std::make_shared<TreeNode>();
        mlp->name = "MLP (SwiGLU)";
        mlp->layer_id = i;
        mlp->type = LayerType::Mlp;
        mlp->depth = 2;
        mlp->parent = layer_item;
        layer_item->children.push_back(mlp);

        auto norm = std::make_shared<TreeNode>();
        norm->name = "RMSNorm";
        norm->layer_id = i;
        norm->type = LayerType::RmsNorm;
        norm->depth = 2;
        norm->parent = layer_item;
        layer_item->children.push_back(norm);

        layers->children.push_back(layer_item);
    }
    tree_root->children.push_back(layers);

    // 3. norm
    auto norm = std::make_shared<TreeNode>();
    norm->name = "norm";
    norm->type = LayerType::RmsNorm;
    norm->depth = 0;
    norm->parent = tree_root;
    tree_root->children.push_back(norm);

    // 4. output
    auto output = std::make_shared<TreeNode>();
    output->name = "output";
    output->type = LayerType::Output;
    output->depth = 0;
    output->parent = tree_root;
    tree_root->children.push_back(output);

    update_visible_nodes();
}

static void get_visible_nodes_rec(const std::shared_ptr<TreeNode>& node, std::vector<std::shared_ptr<TreeNode>>& res) {
    for (auto& child : node->children) {
        res.push_back(child);
        if (child->expandable && child->expanded) {
            get_visible_nodes_rec(child, res);
        }
    }
}

void AppState::update_visible_nodes() {
    std::shared_ptr<TreeNode> selected_node = nullptr;
    if (selected_node_idx >= 0 && selected_node_idx < static_cast<int>(visible_nodes.size())) {
        selected_node = visible_nodes[selected_node_idx];
    }

    visible_nodes.clear();
    get_visible_nodes_rec(tree_root, visible_nodes);

    // Restore selected node index if possible
    if (selected_node) {
        auto it = std::find(visible_nodes.begin(), visible_nodes.end(), selected_node);
        if (it != visible_nodes.end()) {
            selected_node_idx = static_cast<int>(std::distance(visible_nodes.begin(), it));
        } else {
            selected_node_idx = std::clamp(selected_node_idx, 0, static_cast<int>(visible_nodes.size()) - 1);
        }
    } else {
        selected_node_idx = 0;
    }
}

} // namespace llm_tui
