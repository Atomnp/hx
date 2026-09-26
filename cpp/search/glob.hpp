#pragma once

#include <string_view>

namespace hx {

// Shell-style glob matching on '/'-separated paths.
//   *      any run of characters except '/'
//   **     any run of characters including '/'  ("**/" may also match zero directories)
//   ?      one character except '/'
//   [abc]  one of a, b, c   [a-z] a range   [!a] / [^a] anything but a
// Everything else matches itself.
bool glob_match(std::string_view pattern, std::string_view text);

}  // namespace hx
