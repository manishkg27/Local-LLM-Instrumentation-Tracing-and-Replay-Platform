#include "tui/AppState.hpp"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <ctime>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

namespace llm_tui {

// --- JSON Serialization for TelemetryPacket ---
void to_json(json& j, const TelemetryPacket& p) {
    j = json{
        {"sequence_id", p.sequence_id},
        {"timestamp_ns", p.timestamp_ns},
        {"kind", static_cast<int>(p.kind)},
        {"layer_id", p.layer_id},
        {"layer_type", static_cast<int>(p.layer_type)},
        {"device", p.device},
        {"shape", p.shape},
        {"latency_us", p.latency_us},
        {"sparsity", p.sparsity},
        {"mean", p.mean},
        {"sigma", p.sigma},
        {"max_abs", p.max_abs},
        {"anomaly_code", static_cast<int>(p.anomaly_code)},
        {"severity", static_cast<int>(p.severity)}
    };
    j["attn_patch"] = std::vector<float>(std::begin(p.attn_patch), std::end(p.attn_patch));
}

void from_json(const json& j, TelemetryPacket& p) {
    j.at("sequence_id").get_to(p.sequence_id);
    j.at("timestamp_ns").get_to(p.timestamp_ns);
    int kind; j.at("kind").get_to(kind); p.kind = static_cast<PacketKind>(kind);
    j.at("layer_id").get_to(p.layer_id);
    int type; j.at("layer_type").get_to(type); p.layer_type = static_cast<LayerType>(type);
    j.at("device").get_to(p.device);
    j.at("shape").get_to(p.shape);
    j.at("latency_us").get_to(p.latency_us);
    j.at("sparsity").get_to(p.sparsity);
    j.at("mean").get_to(p.mean);
    j.at("sigma").get_to(p.sigma);
    j.at("max_abs").get_to(p.max_abs);
    int acode; j.at("anomaly_code").get_to(acode); p.anomaly_code = static_cast<AnomalyCode>(acode);
    int sev; j.at("severity").get_to(sev); p.severity = static_cast<Severity>(sev);
    if (j.contains("attn_patch")) {
        std::vector<float> patch = j.at("attn_patch").get<std::vector<float>>();
        for (size_t i = 0; i < std::min(patch.size(), size_t(16)); ++i) {
            p.attn_patch[i] = patch[i];
        }
    }
}

// --- JSON Serialization for LedgerEntry ---
void to_json(json& j, const LedgerEntry& e) {
    j = json{
        {"severity", static_cast<int>(e.severity)},
        {"code", static_cast<int>(e.code)},
        {"layer_id", e.layer_id},
        {"timestamp_ns", e.timestamp_ns},
        {"message", e.message}
    };
}

void from_json(const json& j, LedgerEntry& e) {
    int sev; j.at("severity").get_to(sev); e.severity = static_cast<Severity>(sev);
    int code; j.at("code").get_to(code); e.code = static_cast<AnomalyCode>(code);
    j.at("layer_id").get_to(e.layer_id);
    j.at("timestamp_ns").get_to(e.timestamp_ns);
    j.at("message").get_to(e.message);
}

// --- JSON Serialization for ModelTopology ---
void to_json(json& j, const ModelTopology& t) {
    j = json{
        {"n_layer", t.n_layer},
        {"n_head", t.n_head},
        {"n_embd", t.n_embd},
        {"n_vocab", t.n_vocab},
        {"n_params", t.n_params},
        {"name", t.name}
    };
}

void from_json(const json& j, ModelTopology& t) {
    j.at("n_layer").get_to(t.n_layer);
    j.at("n_head").get_to(t.n_head);
    j.at("n_embd").get_to(t.n_embd);
    j.at("n_vocab").get_to(t.n_vocab);
    j.at("n_params").get_to(t.n_params);
    j.at("name").get_to(t.name);
}

AppState::AppState() {
    tree_root = std::make_shared<TreeNode>();
    tree_root->name = "Root";
    tree_root->depth = -1;
    build_tree();
}

void AppState::update_from_packet(const TelemetryPacket& pkt) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    
    // Store packet and cap at 500
    packets.push_back(pkt);
    if (packets.size() > 500) {
        packets.erase(packets.begin(), packets.begin() + (packets.size() - 500));
    }

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

bool AppState::save_to_file(const std::string& filename) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    try {
        json j;
        
        // Metadata
        j["model_name"] = topology.name;
        
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        std::stringstream ss;
        ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
        j["captured_at"] = ss.str();
        
        j["prompt"] = current_prompt;
        
        // Data
        j["model_loaded"] = model_loaded;
        j["topology"] = topology;
        j["packets"] = packets;
        j["anomalies"] = anomalies;
        
        std::ofstream out(filename);
        if (!out) return false;
        out << j.dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

bool AppState::load_from_file(const std::string& filename) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    try {
        std::ifstream in(filename);
        if (!in) return false;
        json j;
        in >> j;
        
        if (j.contains("prompt")) j.at("prompt").get_to(current_prompt);
        if (j.contains("model_loaded")) j.at("model_loaded").get_to(model_loaded);
        if (model_loaded && j.contains("topology")) {
            j.at("topology").get_to(topology);
        }
        if (j.contains("packets")) j.at("packets").get_to(packets);
        if (j.contains("anomalies")) j.at("anomalies").get_to(anomalies);
        
        build_tree();
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace llm_tui
