#pragma once

#include <string>
#include <vector>

namespace hx {

struct SandboxOptions {
    std::vector<std::string> writable;     // directories the command may write to (realpaths)
    std::vector<std::string> deny_read;    // directories it may not even read (secrets)
    bool allow_network = false;
};

// Build a macOS Seatbelt (SBPL) profile: deny by default; allow process/exec, reading files, and
// writing only inside `writable`; block network unless allowed; block reading `deny_read`.
std::string build_profile(const SandboxOptions& opts);

// Apply the profile to the CURRENT process (call it in the child after fork, before exec).
// Everything the process execs afterwards inherits the sandbox and can never leave it.
// Returns false and fills `error` if the platform has no sandbox or the profile is rejected.
bool apply_sandbox(const std::string& profile, std::string& error);

}  // namespace hx
