// Unit tests for OutputCapture. (Process behavior is tested from Python: tests/test_bash.py.)

#include <iostream>
#include <string>

#include "capture.hpp"

static int failures = 0;

#define CHECK(expr)                                                                  \
    do {                                                                             \
        if (!(expr)) {                                                               \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #expr "\n"; \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

int main() {
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
