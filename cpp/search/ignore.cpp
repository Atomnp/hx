#include "ignore.hpp"

#include <fstream>

#include "glob.hpp"

namespace hx {

void IgnoreRules::add_line(std::string_view line, std::string_view base) {
    // Trim trailing whitespace and carriage returns (Windows line endings).
    while (!line.empty() && (line.back() == ' ' || line.back() == '\r' || line.back() == '\t')) line.remove_suffix(1);
    if (line.empty() || line.front() == '#') return;

    IgnoreRule rule;
    rule.base = base;
    if (line.front() == '!') {
        rule.negate = true;
        line.remove_prefix(1);
    }
    if (!line.empty() && line.back() == '/') {
        rule.dir_only = true;
        line.remove_suffix(1);
    }
    if (!line.empty() && line.front() == '/') {
        rule.anchored = true;
        line.remove_prefix(1);
    }
    if (line.find('/') != std::string_view::npos) rule.anchored = true;
    if (line.empty()) return;

    rule.pattern = line;
    rules_.push_back(std::move(rule));
}

void IgnoreRules::add_file(const std::filesystem::path& gitignore, std::string_view base) {
    std::ifstream in(gitignore);
    std::string line;
    while (std::getline(in, line)) add_line(line, base);
}

bool IgnoreRules::ignored(std::string_view rel_path, bool is_dir) const {
    bool result = false;
    for (const auto& rule : rules_) {
        if (rule.dir_only && !is_dir) continue;

        // Rules only see paths below the directory their .gitignore lives in.
        std::string_view sub = rel_path;
        if (!rule.base.empty()) {
            if (sub.size() <= rule.base.size() || sub.substr(0, rule.base.size()) != rule.base ||
                sub[rule.base.size()] != '/')
                continue;
            sub.remove_prefix(rule.base.size() + 1);
        }

        bool hit;
        if (rule.anchored) {
            hit = glob_match(rule.pattern, sub);
        } else {
            size_t slash = sub.rfind('/');
            std::string_view name = slash == std::string_view::npos ? sub : sub.substr(slash + 1);
            hit = glob_match(rule.pattern, name);
        }
        if (hit) result = !rule.negate;
    }
    return result;
}

}  // namespace hx
