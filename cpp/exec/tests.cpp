// Unit tests for OutputCapture and the sandbox profile builder.
// (Process and sandbox behavior is tested end to end from Python: tests/test_bash.py, tests/test_sandbox.py.)

#include <iostream>
#include <string>

#include "capture.hpp"
#include "sandbox.hpp"

static int failures = 0;

#define CHECK(expr)                                                                  \
    do {                                                                             \
        if (!(expr)) {                                                               \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #expr "\n"; \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

static bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

static void test_profile() {
    hx::SandboxOptions o;
    o.writable = {"/work/my proj", "/private/tmp"};
    o.deny_read = {"/Users/me/.ssh"};
    std::string p = hx::build_profile(o);
    CHECK(has(p, "(deny default)"));
    CHECK(has(p, "(subpath \"/work/my proj\")"));
    CHECK(has(p, "(subpath \"/private/tmp\")"));
    CHECK(has(p, "(allow network* (local unix))"));      // no internet by default
    CHECK(!has(p, "(allow network*)\n"));
    CHECK(p.rfind("(deny file-read* (subpath \"/Users/me/.ssh\"))") > p.find("(allow file-read*)"));  // deny comes later, so it wins

    o.allow_network = true;
    CHECK(has(hx::build_profile(o), "(allow network*)\n"));

    o.writable = {"/tmp/we\"ird"};
    CHECK(has(hx::build_profile(o), "(subpath \"/tmp/we\\\"ird\")"));  // quotes escaped
}

int main() {
    test_profile();
    {  // fits: kept whole
        hx::OutputCapture c(100);
        c.append("hello\n", 6);
        CHECK(c.text() == "hello\n");
        CHECK(c.truncated_bytes() == 0);
    }
    {  // overflow: head and tail kept, middle dropped
        hx::OutputCapture c(10);  // 5 head + 5 tail
        std::string s = "AAAAAmiddle-partZZZZZ";
        c.append(s.data(), s.size());
        CHECK(c.total_bytes() == s.size());
        CHECK(c.truncated_bytes() == s.size() - 10);
        std::string t = c.text();
        CHECK(t.rfind("AAAAA", 0) == 0);
        CHECK(t.size() >= 5 && t.substr(t.size() - 5) == "ZZZZZ");
        CHECK(t.find("truncated") != std::string::npos);
    }
    {  // many small appends behave like one big one
        hx::OutputCapture c(20);
        for (int i = 0; i < 10000; ++i) c.append("0123456789", 10);
        CHECK(c.total_bytes() == 100000);
        CHECK(c.truncated_bytes() == 100000 - 20);
        std::string t = c.text();
        CHECK(t.substr(t.size() - 10) == "0123456789");
    }
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all hx-exec tests passed\n";
    return 0;
}
