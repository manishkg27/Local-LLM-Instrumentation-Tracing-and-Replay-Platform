import re

with open('include/tui/AppState.hpp', 'r') as f:
    content = f.read()

if 'std::string generated_text;' not in content:
    content = content.replace('std::recursive_mutex mutex;', 'std::recursive_mutex mutex;\n    \n    std::string current_prompt;\n    std::string generated_text;\n')

with open('include/tui/AppState.hpp', 'w') as f:
    f.write(content)

