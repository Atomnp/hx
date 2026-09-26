#pragma once

#include <string>
#include <string_view>

namespace hx {

// Find a plain substring that every match of the regex must contain, so we can skip lines (and whole
// files) cheaply before running the slow regex engine. Returns "" if there's no safe literal of at
// least 3 characters. Examples:
//   raise \w+Error\(   -> "raise "   (the first of the longest required runs)
//   abc?def            -> "def"      ('c' is optional)
//   foo|bar            -> ""         (alternation: neither side is required)
std::string required_literal(std::string_view regex);

}  // namespace hx
