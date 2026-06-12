#pragma once

#include <ftxui/component/component.hpp>
#include <memory>
#include "tui/AppState.hpp"

namespace llm_tui {

ftxui::Component CreatePanelMetrics(std::weak_ptr<AppState> state);

} // namespace llm_tui
