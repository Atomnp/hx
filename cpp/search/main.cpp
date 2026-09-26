// hx-search: fast code search for the hx harness.
//
//   hx-search grep  PATTERN [ROOT] [--glob G] [-i] [--fixed] [--max N] [--threads N] [--json]
//   hx-search files [ROOT] [--glob G] [--max N] [--json]
//
// Walks ROOT (default "."), honoring .gitignore files, then searches the files on a thread pool.
// Exit codes follow grep: 0 = found something, 1 = found nothing, 2 = error.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "glob.hpp"
#include "ignore.hpp"
#include "json.hpp"
#include "literal.hpp"

namespace fs = std::filesystem;

namespace {

constexpr const char* kUsage =
    "usage: hx-search grep PATTERN [ROOT] [--glob G] [-i] [--fixed] [--max N] [--threads N] [--json]\n"
    "       hx-search files [ROOT] [--glob G] [--max N] [--json]\n";

constexpr size_t kMaxLineChars = 300;

// Directories nobody wants searched, even in a project without a .gitignore.
const std::vector<std::string_view> kAlwaysSkip = {".git", "node_modules", "__pycache__", ".venv",
                                                   "venv", ".mypy_cache", ".pytest_cache"};

struct Options {
    std::string mode;
    std::string pattern;
    std::string root = ".";
    std::string glob;
    bool icase = false;
    bool fixed = false;
    bool json = false;
    size_t max = 200;
    unsigned threads = 0;
};

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << "hx-search: " << msg << "\n" << kUsage;
    std::exit(2);
}

Options parse_args(int argc, char** argv) {
    if (argc < 2) fail("missing mode");
    Options o;
    o.mode = argv[1];
    if (o.mode != "grep" && o.mode != "files") fail("unknown mode '" + o.mode + "'");

    std::vector<std::string> positional;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) fail(a + " needs a value");
            return argv[++i];
        };
        if (a == "--glob") o.glob = value();
        else if (a == "-i" || a == "--ignore-case") o.icase = true;
        else if (a == "--fixed") o.fixed = true;
        else if (a == "--json") o.json = true;
        else if (a == "--max") o.max = std::stoul(value());
        else if (a == "--threads") o.threads = static_cast<unsigned>(std::stoul(value()));
        else if (a.size() > 1 && a[0] == '-') fail("unknown option " + a);
        else positional.push_back(a);
    }

    if (o.mode == "grep") {
        if (positional.empty()) fail("grep needs a PATTERN");
        o.pattern = positional[0];
        if (positional.size() > 1) o.root = positional[1];
    } else if (!positional.empty()) {
        o.root = positional[0];
    }
    while (o.root.size() > 1 && o.root.back() == '/') o.root.pop_back();
    return o;
}

// ---------------------------------------------------------------- walking

struct FileEntry {
    std::string display;  // path as printed: relative to the caller's cwd, like the ROOT they passed
    fs::path full;
};

bool always_skip(std::string_view name) {
    return std::find(kAlwaysSkip.begin(), kAlwaysSkip.end(), name) != kAlwaysSkip.end();
}

bool glob_filter(const std::string& glob, const std::string& rel) {
    if (glob.empty()) return true;
    if (glob.find('/') != std::string::npos) return hx::glob_match(glob, rel);
    size_t slash = rel.rfind('/');
    return hx::glob_match(glob, slash == std::string::npos ? rel : std::string_view(rel).substr(slash + 1));
}

// Depth-first walk. Each directory's .gitignore applies only while we're inside it:
// remember how many rules existed on entry, and truncate back on exit.
void walk(const fs::path& dir, const std::string& rel_dir, const Options& o, hx::IgnoreRules& rules,
          std::vector<FileEntry>& out) {
    size_t mark = rules.size();
    std::error_code ec;
    if (fs::is_regular_file(dir / ".gitignore", ec)) rules.add_file(dir / ".gitignore", rel_dir);

    std::vector<fs::directory_entry> entries;
    for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
         it.increment(ec)) {
        entries.push_back(*it);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.path().filename() < b.path().filename(); });

    for (const auto& e : entries) {
        std::string name = e.path().filename().string();
        std::string rel = rel_dir.empty() ? name : rel_dir + "/" + name;
        // Don't follow symlinked directories: they can form cycles.
        bool is_dir = e.is_directory(ec) && !e.is_symlink(ec);
        if (is_dir) {
            if (always_skip(name) || rules.ignored(rel, true)) continue;
            walk(e.path(), rel, o, rules, out);
        } else if (e.is_regular_file(ec)) {
            if (rules.ignored(rel, false) || !glob_filter(o.glob, rel)) continue;
            out.push_back({o.root == "." ? rel : o.root + "/" + rel, e.path()});
        }
    }
    rules.truncate(mark);
}

std::vector<FileEntry> collect_files(const Options& o) {
    std::vector<FileEntry> files;
    std::error_code ec;
    fs::path root(o.root);
    if (fs::is_regular_file(root, ec)) {
        files.push_back({o.root, root});  // searching a single file
    } else if (fs::is_directory(root, ec)) {
        hx::IgnoreRules rules;
        walk(root, "", o, rules, files);
    } else {
        fail("no such file or directory: " + o.root);
    }
    return files;
}

// ---------------------------------------------------------------- reading

// Maps a file into memory read-only. The OS pages it in as we scan, with no copy into our own buffer.
// RAII: the destructor unmaps and closes, whatever path we leave by.
class MappedFile {
public:
    explicit MappedFile(const fs::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return;
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size == 0) return;
        size_ = static_cast<size_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p != MAP_FAILED) data_ = static_cast<const char*>(p);
    }
    ~MappedFile() {
        if (data_) ::munmap(const_cast<char*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    std::string_view view() const { return data_ ? std::string_view(data_, size_) : std::string_view(); }

private:
    int fd_ = -1;
    const char* data_ = nullptr;
    size_t size_ = 0;
};

bool looks_binary(std::string_view data) {
    return data.substr(0, 8192).find('\0') != std::string_view::npos;
}

// ---------------------------------------------------------------- matching

bool has_regex_syntax(std::string_view p) {
    return p.find_first_of(".^$|?*+()[]{}\\") != std::string_view::npos;
}

std::string lowercase(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// Plain-text patterns use a substring search (much faster than std::regex); everything else uses ECMAScript regex.
// For regexes we also extract a literal every match must contain (`required_`) and check it first:
// std::regex is slow, and most lines (and most files) can be rejected by a cheap substring search.
class Matcher {
public:
    explicit Matcher(const Options& o) : icase_(o.icase) {
        literal_ = o.fixed || !has_regex_syntax(o.pattern);
        if (literal_) {
            needle_ = icase_ ? lowercase(o.pattern) : o.pattern;
        } else {
            required_ = hx::required_literal(o.pattern);
            if (icase_) required_ = lowercase(required_);
            auto flags = std::regex::ECMAScript | std::regex::optimize;
            if (icase_) flags |= std::regex::icase;
            try {
                re_.emplace(o.pattern, flags);
            } catch (const std::regex_error& e) {
                fail("invalid regex '" + o.pattern + "': " + e.what());
            }
        }
    }

    // Cheap whole-file check: if the file can't contain a match, skip it without splitting lines.
    bool file_may_match(std::string_view data) const {
        if (icase_) return true;  // would need a case-folded copy of the file; not worth it
        const std::string& lit = literal_ ? needle_ : required_;
        return lit.empty() || data.find(lit) != std::string_view::npos;
    }

    // `scratch` is per-thread reusable memory for case folding, so matching never allocates per line.
    bool matches(std::string_view line, std::string& scratch) const {
        if (literal_) return contains(line, needle_, scratch);
        if (!required_.empty() && !contains(line, required_, scratch)) return false;
        return std::regex_search(line.data(), line.data() + line.size(), *re_);
    }

private:
    bool contains(std::string_view line, const std::string& lit, std::string& scratch) const {
        if (!icase_) return line.find(lit) != std::string_view::npos;
        scratch.assign(line);
        for (char& c : scratch) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return scratch.find(lit) != std::string::npos;
    }

    bool icase_;
    bool literal_ = false;
    std::string needle_;
    std::string required_;
    std::optional<std::regex> re_;
};

struct Match {
    size_t line;
    std::string text;
};

// ---------------------------------------------------------------- output

std::string clip(std::string_view line) {
    if (line.size() <= kMaxLineChars) return std::string(line);
    return std::string(line.substr(0, kMaxLineChars)) + "...";
}

// ---------------------------------------------------------------- modes

int run_files(const Options& o, const std::vector<FileEntry>& files) {
    size_t shown = std::min(files.size(), o.max);
    for (size_t i = 0; i < shown; ++i) {
        if (o.json) std::cout << "{\"path\":\"" << hx::json_escape(files[i].display) << "\"}\n";
        else std::cout << files[i].display << "\n";
    }
    if (o.json) {
        std::cout << "{\"summary\":{\"files\":" << files.size() << ",\"truncated\":"
                  << (files.size() > o.max ? "true" : "false") << "}}\n";
    }
    return files.empty() ? 1 : 0;
}

int run_grep(const Options& o, const std::vector<FileEntry>& files) {
    const Matcher matcher(o);
    std::vector<std::vector<Match>> results(files.size());  // one slot per file: no locking needed
    std::atomic<size_t> next{0};
    std::atomic<size_t> found{0};

    // Work stealing in its simplest form: each thread grabs the next file index until none are left.
    auto worker = [&] {
        std::string scratch;
        for (;;) {
            size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= files.size() || found.load(std::memory_order_relaxed) >= o.max) return;

            MappedFile file(files[i].full);
            std::string_view data = file.view();
            if (data.empty() || looks_binary(data) || !matcher.file_may_match(data)) continue;

            size_t line_no = 1;
            for (size_t pos = 0; pos < data.size(); ++line_no) {
                size_t nl = data.find('\n', pos);
                size_t end = nl == std::string_view::npos ? data.size() : nl;
                std::string_view line = data.substr(pos, end - pos);
                if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

                if (matcher.matches(line, scratch)) {
                    results[i].push_back({line_no, clip(line)});
                    if (found.fetch_add(1, std::memory_order_relaxed) + 1 >= o.max) return;
                }
                if (nl == std::string_view::npos) break;
                pos = nl + 1;
            }
        }
    };

    unsigned n = o.threads ? o.threads : std::max(1u, std::thread::hardware_concurrency());
    n = std::min<unsigned>(n, std::max<size_t>(1, files.size()));
    {
        std::vector<std::jthread> pool;  // jthread joins in its destructor
        for (unsigned t = 0; t < n; ++t) pool.emplace_back(worker);
    }

    // Print in file order so output is deterministic regardless of thread timing.
    size_t printed = 0;
    for (size_t i = 0; i < files.size() && printed < o.max; ++i) {
        for (const auto& m : results[i]) {
            if (printed++ >= o.max) break;
            if (o.json) {
                std::cout << "{\"path\":\"" << hx::json_escape(files[i].display) << "\",\"line\":" << m.line
                          << ",\"text\":\"" << hx::json_escape(m.text) << "\"}\n";
            } else {
                std::cout << files[i].display << ":" << m.line << ":" << m.text << "\n";
            }
        }
    }
    bool truncated = found.load() >= o.max;
    if (o.json) {
        std::cout << "{\"summary\":{\"files\":" << files.size() << ",\"matches\":" << printed
                  << ",\"truncated\":" << (truncated ? "true" : "false") << "}}\n";
    } else if (truncated) {
        std::cerr << "hx-search: stopped at " << o.max << " matches\n";
    }
    return printed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    Options o = parse_args(argc, argv);
    std::vector<FileEntry> files = collect_files(o);
    return o.mode == "grep" ? run_grep(o, files) : run_files(o, files);
}
