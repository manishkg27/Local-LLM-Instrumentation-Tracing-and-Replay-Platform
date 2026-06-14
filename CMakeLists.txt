cmake_minimum_required(VERSION 3.20)
project(llm-tui
    VERSION 0.1.0
    DESCRIPTION "Local LLM Instrumentation, Tracing, and Replay Platform"
    LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS         OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
# llama.cpp is added as a git submodule under third_party/llama.cpp/.
# We build only the C library (no tools/server/examples) and link statically
# against the headers we need.

# Where to look for downloaded GGUF model files.
set(LLM_TUI_MODEL_DIR "/home/manish/models" CACHE PATH "Directory containing *.gguf model files")

# ----------------------------------------------------------------------------
# llama.cpp as a submodule
# ----------------------------------------------------------------------------
# We don't modify llama.cpp — we just build its C library and link against it.
# Disable everything we don't need (server, tools, examples, common, tests).
set(BUILD_SHARED_LIBS               ON  CACHE BOOL "Build llama.cpp + ggml as shared libs" FORCE)
set(LLAMA_BUILD_COMMON              OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS               OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES            OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS               OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER              OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP                 OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI                  OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI           OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL                      OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL                   OFF CACHE BOOL "" FORCE)
# Force CPU-only backends (no CUDA/Metal/etc. detection) for Day 1.
set(GGML_CUDA                       OFF CACHE BOOL "" FORCE)
set(GGML_METAL                      OFF CACHE BOOL "" FORCE)
set(GGML_HIP                        OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN                     OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL                     OFF CACHE BOOL "" FORCE)
set(GGML_SYCL                       OFF CACHE BOOL "" FORCE)
set(GGML_RPC                        OFF CACHE BOOL "" FORCE)
set(GGML_BLAS                       OFF CACHE BOOL "" FORCE)

# Disable noisy warnings from third-party code that we don't own.
set(LLAMA_ALL_WARNINGS              OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS               OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

# ----------------------------------------------------------------------------
# Dependencies via vcpkg
# ----------------------------------------------------------------------------
find_package(ftxui        CONFIG REQUIRED)
find_package(spdlog       CONFIG REQUIRED)
find_package(fmt          CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Catch2 (header-only v3) — vendored differently in vcpkg
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    # vcpkg installs it as a CMake target `Catch2::Catch2WithMain`
    find_package(Catch2 CONFIG QUIET)
    if(NOT Catch2_FOUND)
        # Manual header path (when vcpkg is not used)
        find_path(CATCH2_INCLUDE_DIR
            NAMES catch2/catch_test_macros.hpp
            REQUIRED
        )
        add_library(Catch2::Catch2WithMain INTERFACE IMPORTED)
        target_include_directories(Catch2::Catch2WithMain INTERFACE ${CATCH2_INCLUDE_DIR})
    endif()
endif()

# ----------------------------------------------------------------------------
# Common library (the project's own code, shared by all binaries)
# ----------------------------------------------------------------------------
add_library(llm_tui_core STATIC
    src/core/RingBuffer.cpp
    src/core/AnomalyDetector.cpp
    src/hook/LlamaInterceptor.cpp
)
target_include_directories(llm_tui_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(llm_tui_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src
)
target_link_libraries(llm_tui_core
    PUBLIC
        spdlog::spdlog
        fmt::fmt
        nlohmann_json::nlohmann_json
    PRIVATE
        llama       # exposes llama.h + links to ggml transitively
)
target_compile_features(llm_tui_core PUBLIC cxx_std_20)

# ----------------------------------------------------------------------------
# Day 1 — hello_inference  (Person 1)
# ----------------------------------------------------------------------------
add_executable(hello_inference src/hook/hello_inference.cpp)
target_link_libraries(hello_inference PRIVATE llm_tui_core)

# ----------------------------------------------------------------------------
# TUI — tui_hello  (Person 2)
# ----------------------------------------------------------------------------
add_executable(tui_hello
    src/tui/tui_hello.cpp
    src/tui/AppState.cpp
    src/tui/PanelTopology.cpp
    src/tui/PanelPacketStream.cpp
    src/tui/PanelAttention.cpp
    src/tui/PanelMetrics.cpp
    src/tui/PanelAnomalies.cpp
)
target_link_libraries(tui_hello PRIVATE llm_tui_core ftxui::dom ftxui::component ftxui::screen)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
enable_testing()

add_executable(test_ringbuffer tests/test_ringbuffer.cpp)
target_link_libraries(test_ringbuffer PRIVATE llm_tui_core Catch2::Catch2WithMain)
add_test(NAME test_ringbuffer COMMAND test_ringbuffer)

add_executable(test_anomaly tests/test_anomaly.cpp)
target_link_libraries(test_anomaly PRIVATE llm_tui_core Catch2::Catch2WithMain)
add_test(NAME test_anomaly COMMAND test_anomaly)

# ----------------------------------------------------------------------------
# Install (for packaging)
# ----------------------------------------------------------------------------
install(TARGETS hello_inference tui_hello RUNTIME DESTINATION bin)
