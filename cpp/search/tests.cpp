// Unit tests for glob matching and .gitignore rules. Run with `ctest` (or ./hx-search-tests).
// A tiny CHECK macro instead of a test framework, so there's nothing to install.

#include <iostream>

#include "glob.hpp"
#include "ignore.hpp"
#include "literal.hpp"

static int failures = 0;

#define CHECK(expr)                                                                  \
    do {                                                                             \
        if (!(expr)) {                                                               \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #expr "\n"; \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

using hx::glob_match;

static void test_glob() {
    CHECK(glob_match("*.py", "main.py"));
    CHECK(!glob_match("*.py", "main.pyc"));
    CHECK(!glob_match("*.py", "src/main.py"));  // * doesn't cross '/'
    CHECK(glob_match("src/*.py", "src/main.py"));
    CHECK(glob_match("**/*.py", "a/b/c.py"));
    CHECK(glob_match("**/*.py", "c.py"));        // **/ can match zero directories
    CHECK(glob_match("src/**", "src/a/b.txt"));
    CHECK(glob_match("test_?.py", "test_1.py"));
    CHECK(!glob_match("test_?.py", "test_10.py"));
    CHECK(glob_match("[abc].txt", "b.txt"));
    CHECK(!glob_match("[!abc].txt", "b.txt"));
    CHECK(glob_match("[a-z]*.md", "readme.md"));
    CHECK(glob_match("[x", "[x"));               // unclosed bracket is literal
    CHECK(glob_match("", ""));
    CHECK(!glob_match("", "a"));
}

static void test_ignore() {
    hx::IgnoreRules r;
    r.add_line("# comment", "");
    r.add_line("*.log", "");
    r.add_line("build/", "");
    r.add_line("/dist", "");
    r.add_line("docs/*.tmp", "");
    r.add_line("!keep.log", "");

    CHECK(r.ignored("app.log", false));
    CHECK(r.ignored("deep/nested/app.log", false));  // unanchored: matches the name anywhere
    CHECK(!r.ignored("keep.log", false));            // negation re-includes
    CHECK(r.ignored("build", true));
    CHECK(!r.ignored("build", false));               // "build/" is dir-only: a *file* named build is kept
    CHECK(r.ignored("dist", true));
    CHECK(!r.ignored("src/dist", true));             // anchored to the .gitignore's directory
    CHECK(r.ignored("docs/x.tmp", false));
    CHECK(!r.ignored("other/docs/x.tmp", false));

    // Rules from a nested .gitignore only apply below that directory.
    size_t mark = r.size();
    r.add_line("secret.txt", "pkg");
    CHECK(r.ignored("pkg/secret.txt", false));
    CHECK(!r.ignored("secret.txt", false));
    r.truncate(mark);
    CHECK(!r.ignored("pkg/secret.txt", false));
}

static void test_required_literal() {
    using hx::required_literal;
    CHECK(required_literal(R"(raise \w+Error\()") == "raise ");
    CHECK(required_literal("ConnectionTimeoutError") == "ConnectionTimeoutError");
    CHECK(required_literal("abc?def") == "def");       // 'c' is optional
    CHECK(required_literal("(abc)?xyz1") == "xyz1");   // group contents may not appear
    CHECK(required_literal("foo|bar") == "");          // alternation: nothing is required
    CHECK(required_literal("a.b") == "");              // runs shorter than 3
    CHECK(required_literal(R"(\.config)") == ".config");
    CHECK(required_literal("[abc]+tail") == "tail");
    CHECK(required_literal("x{2,3}yz") == "");         // 'x' optional-ish, "yz" too short
}

int main() {
    test_glob();
    test_ignore();
    test_required_literal();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all hx-search tests passed\n";
    return 0;
}
