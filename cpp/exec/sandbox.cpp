#include "sandbox.hpp"

#include <string_view>

#ifdef __APPLE__
// sandbox_init() is the C API behind `sandbox-exec`. Apple marks it deprecated but still ships it,
// and it's what Chromium and Codex CLI use. Silence the deprecation warning for these two calls only.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <sandbox.h>
#endif

namespace hx {
namespace {

// SBPL is a Scheme dialect; string literals escape \ and ".
std::string sbpl_string(std::string_view s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out + "\"";
}

}  // namespace

std::string build_profile(const SandboxOptions& opts) {
    std::string p =
        "(version 1)\n"
        "(deny default)\n"                     // anything not allowed below is refused by the kernel
        "(allow process-exec)\n"
        "(allow process-fork)\n"
        "(allow process-info*)\n"
        "(allow signal (target same-sandbox))\n"  // may signal its own children, nothing else
        "(allow sysctl-read)\n"
        "(allow mach-lookup)\n"                // system services (DNS, keychain lookups, ...) used by most binaries
        "(allow ipc-posix-shm)\n"
        "(allow file-read*)\n"                 // read anywhere (minus deny_read, below)
        "(allow file-ioctl (regex #\"^/dev/tty\"))\n"
        "(allow file-write*\n"
        "  (literal \"/dev/null\") (literal \"/dev/zero\") (regex #\"^/dev/tty\")";
    for (const auto& dir : opts.writable) p += "\n  (subpath " + sbpl_string(dir) + ")";
    p += ")\n";

    if (opts.allow_network) p += "(allow network*)\n";
    else p += "(allow network* (local unix))\n";  // local unix sockets only; no internet

    // Later rules win in SBPL, so these denies override the broad file-read* allow above.
    for (const auto& dir : opts.deny_read) p += "(deny file-read* (subpath " + sbpl_string(dir) + "))\n";
    return p;
}

bool apply_sandbox(const std::string& profile, std::string& error) {
#ifdef __APPLE__
    char* err = nullptr;
    if (sandbox_init(profile.c_str(), 0, &err) != 0) {
        error = err ? err : "sandbox_init failed";
        if (err) sandbox_free_error(err);
        return false;
    }
    return true;
#else
    (void)profile;
    error = "sandboxing is only implemented for macOS (Linux would use Landlock or bubblewrap)";
    return false;
#endif
}

}  // namespace hx

#ifdef __APPLE__
#pragma clang diagnostic pop
#endif
