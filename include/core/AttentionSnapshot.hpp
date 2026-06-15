// =============================================================================
//  AttentionSnapshot.hpp
//  -----------------------------------------------------------------------------
//  Thread-safe shared buffer for streaming full attention matrices from the
//  llama.cpp interceptor thread to the TUI rendering thread.
//
//  Design:
//    - The producer (on_eval callback) writes the latest full attention matrix
//      for each (layer, head) pair.
//    - The consumer (PanelAttention) reads the most recent snapshot.
//    - A std::shared_mutex allows concurrent reads while serializing writes.
// =============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace llm_tui {

// A single full attention matrix snapshot for one (layer, head) pair.
// Matrix is stored row-major: matrix[row * n_kv + col] = attention(row, col)
// where row = query position, col = key position.
struct AttentionMatrix {
    int layer_id = -1;
    int head_id  = 0;
    int n_kv     = 0;   // number of key positions in cache
    int n_head   = 0;   // total number of attention heads
    int n_tok    = 0;   // number of query tokens
    uint64_t timestamp_ns = 0;
    std::vector<float> data;  // n_tok * n_kv floats, row-major

    void clear() {
        layer_id = -1;
        head_id = 0;
        n_kv = 0;
        n_head = 0;
        n_tok = 0;
        timestamp_ns = 0;
        data.clear();
    }

    bool empty() const { return data.empty() || n_kv == 0 || n_tok == 0; }
};

// Thread-safe attention matrix cache. Stores the latest snapshot for each
// active layer (one at a time, since we're processing sequentially).
// In practice, we only need the latest matrix since the TUI displays the
// most recent attention weights.
class AttentionCache {
public:
    // Producer: write the latest full attention matrix.
    void store(const AttentionMatrix& matrix) {
        std::lock_guard<std::shared_mutex> lock(mtx_);
        current_ = matrix;
    }

    // Consumer: read the latest full attention matrix.
    // Returns a copy to avoid holding the lock during rendering.
    AttentionMatrix load() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        return current_;
    }

    // Check if we have data without copying.
    bool has_data() const {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        return !current_.empty();
    }

    void clear() {
        std::lock_guard<std::shared_mutex> lock(mtx_);
        current_.clear();
    }

private:
    mutable std::shared_mutex mtx_;
    AttentionMatrix current_;
};

} // namespace llm_tui