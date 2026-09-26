#include "literal.hpp"

#include <cctype>

namespace hx {

std::string required_literal(std::string_view re) {
    // With alternation, nothing is guaranteed to appear. Keep the analysis simple and safe.
    if (re.find('|') != std::string_view::npos) return "";

    std::string best, run;
    int depth = 0;  // only literals outside groups are required ("(abc)?" may not appear)

    auto finish = [&] {
        if (run.size() > best.size()) best = run;
        run.clear();
    };
    auto quantifier_follows = [&](size_t i) {
        return i < re.size() && (re[i] == '?' || re[i] == '*' || re[i] == '{');
    };

    for (size_t i = 0; i < re.size(); ++i) {
        char c = re[i];
        if (c == '\\' && i + 1 < re.size()) {
            char e = re[++i];
            if (std::isalnum(static_cast<unsigned char>(e))) {  // \w \d \s \b ... are classes, not literals
                finish();
                continue;
            }
            if (depth == 0) {
                if (quantifier_follows(i + 1)) finish();  // "x\.?" : the escaped char is optional
                else run += e;
            }
            continue;
        }
        switch (c) {
            case '(': finish(); ++depth; break;
            case ')': finish(); if (depth > 0) --depth; break;
            case '[': {  // skip a whole character class
                finish();
                size_t close = re.find(']', i + 2);
                i = close == std::string_view::npos ? re.size() : close;
                break;
            }
            case '?': case '*': case '{':
                // The previous character is optional: it can't be part of the required run.
                if (!run.empty()) run.pop_back();
                finish();
                if (c == '{') {
                    size_t close = re.find('}', i);
                    i = close == std::string_view::npos ? re.size() : close;
                }
                break;
            case '+': finish(); break;  // "a+" still requires one 'a', but the run ends there
            case '.': case '^': case '$': finish(); break;
            default:
                if (depth == 0) run += c;
        }
    }
    finish();
    return best.size() >= 3 ? best : "";
}

}  // namespace hx
