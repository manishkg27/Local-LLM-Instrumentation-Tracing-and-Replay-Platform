# Local LLM Instrumentation, Tracing, and Replay Platform

> A lightweight, non-invasive telemetry + diagnostic TUI for local transformer
> models. Hooks into a running **llama.cpp** model, captures layer-by-layer
> latency, attention matrices, tensor stats, and numerical anomalies, and
> renders them in a **5-panel terminal UI** with **vim-like keybindings** —
> inspired by `lazygit` and `btop`, with NerdFont glyphs.

![Day 1 screenshot](images/day1.png)

---

## ✨ Features

- **Non-invasive instrumentation** — we link against the public `libllama.so`
  C API. **Zero modifications to llama.cpp source.**
- **Layer-by-layer execution latency** — wrap `llama_decode()` with chrono
  timing; map ggml graph nodes to layers.
- **Attention matrix visualisation** — ` ░▒▓█` NerdFont heatmap, pannable,
  contrast-adjustable.
- **Runtime metrics inspection** — tensor shape, dtype, sparsity, mean, max.
- **Numerical anomaly detection** — outlier features, clipping risk, dead
  layers, latency hotspots, NaN/Inf.
- **Bounded ring buffer** — cap RAM, back-pressure when full.
- **5 panels**: Model Topology · Live Packet Stream · Attention Matrix ·
  Runtime Metrics · Anomaly Ledger.
- **Vim keybindings**: `h j k l` nav, `Tab` cycle focus, `+ / -` adjust,
  `Space` toggle, `g / G` top/bottom, `q` quit.

---

## 🧱 Tech stack

| Component        | Library            | Why |
|------------------|--------------------|-----|
| TUI              | **FTXUI v5+**      | Header-only, NerdFont-aware, declarative, perfect for 5-panel UIs |
| LLM runtime      | **llama.cpp**      | The project requirement; C API is stable |
| Graph introspect | **ggml** (bundled) | Hook compute-graph nodes for tensor stats |
| Logging          | **spdlog**         | Async sinks, severity icons, plays nice with FTXUI |
| Formatting       | **fmt**            | Used by spdlog + llama.cpp already — zero extra cost |
| Unit tests       | **Catch2 v3**      | BDD-style, single-header option |
| Build / deps     | **CMake 3.20+ + vcpkg** | `vcpkg.json` pins versions, CI-friendly |
| JSON (config)    | **nlohmann/json**  | For `config.json` / replay records |
| Ring buffer      | `std::deque` + `std::mutex` + `std::condition_variable` | Beginner-friendly, correct, plenty fast |
| Timing           | `std::chrono`      | Built-in |

---

## 📦 Prerequisites

You need:

- **CMake ≥ 3.20**           (we tested with 4.3.3)
- **GCC ≥ 13** or **Clang ≥ 16** with C++20 support
- **vcpkg**                  (one-time bootstrap)
- **A pre-built llama.cpp**  (clone + `cmake -B build && cmake --build build -j`)
- **A NerdFont** in your terminal  (e.g. JetBrainsMono Nerd Font)
- **A GGUF model**, e.g. Qwen2.5-Coder-3B-Instruct Q4_K_M (~2 GB)

### Arch Linux quick install

```bash
# Toolchain
sudo pacman -S --needed base-devel cmake git wget

# vcpkg (one-time)
git clone --depth 1 https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
echo 'export VCPKG_ROOT=$HOME/vcpkg' >> ~/.bashrc
echo 'export PATH=$VCPKG_ROOT:$PATH'        >> ~/.bashrc

# Nerd Font
yay -S ttf-jetbrains-mono-nerd
# OR sudo pacman -S ttf-jetbrains-mono-nerd

# llama.cpp (pre-built shared libs at /home/manish/llama.cpp)
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DBUILD_SHARED_LIBS=ON
cmake --build ~/llama.cpp/build -j$(nproc)

# Model
mkdir -p ~/models
wget -c -O ~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

---

## 🏗️ Build

```bash
cd /home/manish/llm-tui
cmake -B build -S . \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

This will:
1. Run `vcpkg` to install `ftxui`, `spdlog`, `fmt`, `catch2`, `nlohmann-json`.
2. Configure CMake against the pre-built `~/llama.cpp` shared libs.
3. Build `hello_inference`, `tui_hello`, and `test_ringbuffer`.

---

## 🚀 Run

### Day 1 — Hello, world (Person 1)

```bash
./build/hello_inference
# or specify a different model
./build/hello_inference /path/to/your-model.gguf
```

Expected output: a 3-line summary of the model's topology, plus
`llama_decode` latency for the prompt "Hello".

### Day 1 — TUI hello (Person 2)

```bash
./build/tui_hello
```

You should see a NerdFont-bordered header, **5 mock panels** (one per
future day), and a footer showing live `keypresses=…`. Try:

```
j  j  k  l  Tab  h  q
```

`q` (or `Esc`) quits.

---

## 🗺️ Roadmap (8-day plan)

| Day | Date       | Theme | Deliverable |
|-----|------------|-------|-------------|
| 1   | Mon 8 Jun  | Foundation         | Build env, TelemetryPacket, hello_inference + tui_hello |
| 2   | Tue 9 Jun  | Core infra        | RingBuffer + tests, 5-panel grid + Tab focus |
| 3   | Wed 10 Jun | Hooks → P1        | LlamaInterceptor → Panel 1 (Model Topology) |
| 4   | Thu 11 Jun | Stats + P2/P3     | Tensor stats → Panel 2 (Packet Stream) + Panel 3 (Attention) |
| 5   | Fri 12 Jun | Anomaly + P4/P5   | AnomalyDetector → Panel 4 (Metrics) + Panel 5 (Anomaly Ledger) |
| 6   | Sat 13 Jun | Integration + UX  | End-to-end, consumer thread, vim command bar, help overlay |
| 7   | Sun 14 Jun | Integration test  | 256-token run on Qwen2.5-Coder, bug bash |
| 8   | Mon 15 Jun | 🛟 BUFFER          | README, demo GIF, cross-platform check, polish |
| —   | Tue 16 Jun | **SUBMIT**         | 🎯 |

---

## 🗂️ Project layout

```
llm-tui/
├── CMakeLists.txt
├── vcpkg.json
├── README.md
├── include/
│   ├── core/
│   │   ├── TelemetryPacket.hpp   # 80B POD, the data flowing through the system
│   │   ├── RingBuffer.hpp        # thread-safe bounded FIFO
│   │   └── AnomalyDetector.hpp   # numerical-anomaly rules
│   └── hook/
│       └── LlamaInterceptor.hpp  # thin RAII wrapper around llama.cpp C API
├── src/
│   ├── core/      (RingBuffer.cpp, AnomalyDetector.cpp)
│   ├── hook/      (LlamaInterceptor.cpp, hello_inference.cpp)
│   └── tui/       (tui_hello.cpp)
├── tests/
│   └── test_ringbuffer.cpp       # Catch2 v3 tests, run via `ctest`
├── docs/                         # architecture.md, keybindings.md, etc.
├── images/                       # screenshots, demo GIFs
└── third_party/                  # reserved (we link against ~/llama.cpp instead)
```

---

## ⌨️ Default keybindings (lazygit / btop inspired)

| Key            | Action |
|----------------|--------|
| `Tab` / `Shift+Tab` | Cycle focus across 5 panels |
| `h` / `l`      | Previous / next panel (vim-style) |
| `j` / `k`      | Down / up in the focused panel |
| `g` / `G`      | Jump to top / bottom of list |
| `Space`        | Expand / collapse / freeze-scroll |
| `+` / `-`      | Adjust contrast / progress / zoom |
| `:`            | Open command bar (`:q` quits) |
| `?`            | Toggle help overlay |
| `q` / `Esc`    | Quit |

---

