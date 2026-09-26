#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace hx {

// One line of a .gitignore file.
struct IgnoreRule {
    std::string pattern;   // glob, without the leading '!' / '/' and trailing '/'
    std::string base;      // directory (relative to the search root) of the .gitignore it came from
    bool negate = false;   // "!pattern" re-includes something an earlier rule ignored
    bool dir_only = false; // "pattern/" only matches directories
    bool anchored = false; // contains a '/' (other than at the end): matched against the path, not the name
};

// The .gitignore rules in effect at some point of a directory walk.
// The walker pushes a directory's rules when it enters and truncates back when it leaves,
// so rules from a/.gitignore never apply inside b/.
class IgnoreRules {
public:
    void add_line(std::string_view line, std::string_view base);
    void add_file(const std::filesystem::path& gitignore, std::string_view base);

    // `rel_path` is relative to the search root, '/'-separated. The last matching rule wins.
    bool ignored(std::string_view rel_path, bool is_dir) const;

    size_t size() const { return rules_.size(); }
    void truncate(size_t n) { rules_.resize(n); }

private:
    std::vector<IgnoreRule> rules_;
};

}  // namespace hx
