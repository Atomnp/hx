#include "glob.hpp"

namespace hx {
namespace {

// Try to match a [...] character class at the start of `p` against `c`.
// On success returns true and sets `consumed` to the class length (including brackets).
// If `p` has no closing bracket, the '[' is treated as a literal by the caller.
bool match_class(std::string_view p, char c, bool& matched, size_t& consumed) {
    // A ']' right after '[' (or after '[!') is a literal member, so search for the close from there.
    size_t start = 1;
    bool negate = start < p.size() && (p[start] == '!' || p[start] == '^');
    if (negate) ++start;
    size_t close = p.find(']', start + 1);
    if (close == std::string_view::npos) return false;

    std::string_view members = p.substr(start, close - start);
    bool hit = false;
    for (size_t i = 0; i < members.size(); ++i) {
        if (i + 2 < members.size() && members[i + 1] == '-') {
            if (c >= members[i] && c <= members[i + 2]) hit = true;
            i += 2;
        } else if (members[i] == c) {
            hit = true;
        }
    }
    matched = hit != negate;
    consumed = close + 1;
    return true;
}

}  // namespace

bool glob_match(std::string_view p, std::string_view t) {
    while (!p.empty()) {
        char pc = p.front();

        if (pc == '*') {
            if (p.size() > 1 && p[1] == '*') {
                p.remove_prefix(2);
                // "**/" can match zero directories: "**/x.py" matches "x.py".
                if (!p.empty() && p.front() == '/' && glob_match(p.substr(1), t)) return true;
                // Otherwise let ** swallow any number of characters, slashes included.
                for (size_t i = 0; i <= t.size(); ++i) {
                    if (glob_match(p, t.substr(i))) return true;
                }
                return false;
            }
            p.remove_prefix(1);
            // A single * stops at the next '/'.
            for (size_t i = 0; i <= t.size(); ++i) {
                if (glob_match(p, t.substr(i))) return true;
                if (i < t.size() && t[i] == '/') break;
            }
            return false;
        }

        if (t.empty()) return false;

        if (pc == '?') {
            if (t.front() == '/') return false;
            p.remove_prefix(1);
            t.remove_prefix(1);
            continue;
        }

        if (pc == '[') {
            bool matched = false;
            size_t consumed = 0;
            if (match_class(p, t.front(), matched, consumed)) {
                if (!matched) return false;
                p.remove_prefix(consumed);
                t.remove_prefix(1);
                continue;
            }
            // No closing bracket: fall through and match '[' literally.
        }

        if (pc != t.front()) return false;
        p.remove_prefix(1);
        t.remove_prefix(1);
    }
    return t.empty();
}

}  // namespace hx
