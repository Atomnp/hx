#pragma once

// Tiny JSON helpers shared by the hx native tools. Header-only: `inline` lets every
// translation unit include it without duplicate-symbol errors at link time.

#include <cstdio>
#include <string>
#include <string_view>

namespace hx {

inline std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

inline std::string json_string(std::string_view s) { return "\"" + json_escape(s) + "\""; }

}  // namespace hx
