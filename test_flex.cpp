#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
int main() {
    using namespace ftxui;
    auto c0 = Renderer([] { return window(text("A"), text("A")) | size(WIDTH, EQUAL, 10); });
    auto c1 = Renderer([] { return window(text("B"), text("B")) | flex; });
    auto c2 = Renderer([] { return window(text("C"), text("C")) | size(WIDTH, EQUAL, 10); });
    auto hc = Container::Horizontal({c0, c1, c2});
    auto screen = ScreenInteractive::TerminalOutput();
    screen.Loop(hc);
}
