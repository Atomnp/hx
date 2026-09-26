// hx-exec: run one command for the hx harness and report what happened, as JSON.
//
//   hx-exec [--timeout SEC] [--max-output BYTES] [--cwd DIR] [--cpu SEC] [--fsize MB]
//           [--sandbox [--allow-write DIR]... [--deny-read DIR]... [--allow-network]] -- CMD [ARGS...]
//
// What it guarantees, and plain subprocess calls don't:
//   * the command can't wait on the keyboard: stdin is /dev/null
//   * it can't run forever: on timeout the whole PROCESS GROUP gets SIGTERM, then SIGKILL
//   * it can't leave children behind: anything still in its group when it exits is killed
//   * it can't flood memory or context: output is capped (head + tail kept)
//   * resource limits (CPU seconds, max file size, no core dumps) are applied before exec
//   * Ctrl-C / SIGTERM sent to hx-exec are forwarded to the command's group
//   * with --sandbox, the kernel itself confines the command (macOS Seatbelt): writes only to the
//     allowed directories, no network unless allowed, secrets unreadable. See sandbox.cpp.
//
// Output: one JSON object on stdout:
//   {"exit_code":0,"signal":null,"timed_out":false,"interrupted":false,"duration_ms":12,
//    "output_bytes":6,"truncated_bytes":0,"output":"hello\n"}

#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "capture.hpp"
#include "json.hpp"
#include "sandbox.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    double timeout_s = 120;
    size_t max_output = 30000;
    std::string cwd;
    long cpu_s = 0;     // 0 = no limit
    long fsize_mb = 0;  // 0 = no limit
    bool sandbox = false;
    hx::SandboxOptions sandbox_opts;
    std::vector<std::string> command;
};

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << "hx-exec: " << msg << "\n"
              << "usage: hx-exec [--timeout SEC] [--max-output BYTES] [--cwd DIR] [--cpu SEC] [--fsize MB]\n"
              << "               [--sandbox [--allow-write DIR]... [--deny-read DIR]... [--allow-network]] -- CMD [ARGS...]\n";
    std::exit(125);
}

Options parse_args(int argc, char** argv) {
    Options o;
    int i = 1;
    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--") {
            ++i;
            break;
        }
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) fail(a + " needs a value");
            return argv[++i];
        };
        if (a == "--timeout") o.timeout_s = std::stod(value());
        else if (a == "--max-output") o.max_output = std::stoul(value());
        else if (a == "--cwd") o.cwd = value();
        else if (a == "--cpu") o.cpu_s = std::stol(value());
        else if (a == "--fsize") o.fsize_mb = std::stol(value());
        else if (a == "--sandbox") o.sandbox = true;
        else if (a == "--allow-write") o.sandbox_opts.writable.push_back(value());
        else if (a == "--deny-read") o.sandbox_opts.deny_read.push_back(value());
        else if (a == "--allow-network") o.sandbox_opts.allow_network = true;
        else fail("unknown option " + a);
    }
    for (; i < argc; ++i) o.command.emplace_back(argv[i]);
    if (o.command.empty()) fail("no command given (put it after --)");
    return o;
}

// Set from a signal handler, so it must be a lock-free type: volatile sig_atomic_t.
volatile std::sig_atomic_t g_forward_signal = 0;

void on_signal(int sig) { g_forward_signal = sig; }

void install_signal_forwarding() {
    struct sigaction sa {};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART: we want poll() to wake up with EINTR
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);
}

void apply_limits(const Options& o) {
    struct rlimit no_core {0, 0};
    setrlimit(RLIMIT_CORE, &no_core);  // a crashing test shouldn't write a multi-GB core file
    if (o.cpu_s > 0) {
        struct rlimit cpu {static_cast<rlim_t>(o.cpu_s), static_cast<rlim_t>(o.cpu_s + 1)};
        setrlimit(RLIMIT_CPU, &cpu);  // SIGXCPU at the soft limit, SIGKILL at the hard limit
    }
    if (o.fsize_mb > 0) {
        rlim_t bytes = static_cast<rlim_t>(o.fsize_mb) * 1024 * 1024;
        struct rlimit fsize {bytes, bytes};
        setrlimit(RLIMIT_FSIZE, &fsize);  // writing past it raises SIGXFSZ
    }
}

// Runs in the forked child. Only async-signal-safe calls between fork() and exec() in a threaded parent;
// hx-exec is single-threaded, but we keep to that rule anyway.
[[noreturn]] void exec_child(const Options& o, const std::string& profile, int out_fd, char* const* argv) {
    setpgid(0, 0);  // new process group: lets the parent signal the command AND everything it spawns

    int devnull = open("/dev/null", O_RDONLY);
    dup2(devnull, STDIN_FILENO);
    dup2(out_fd, STDOUT_FILENO);
    dup2(out_fd, STDERR_FILENO);  // merged, so stdout and stderr stay in the order they were written
    close(devnull);
    close(out_fd);

    // Undo our signal handlers in the child.
    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGHUP, SIG_DFL);

    if (!o.cwd.empty() && chdir(o.cwd.c_str()) != 0) {
        dprintf(STDERR_FILENO, "hx-exec: cannot cd to %s: %s\n", o.cwd.c_str(), strerror(errno));
        _exit(126);
    }
    apply_limits(o);
    if (o.sandbox) {
        std::string error;
        if (!hx::apply_sandbox(profile, error)) {
            dprintf(STDERR_FILENO, "hx-exec: sandbox failed: %s\n", error.c_str());
            _exit(126);  // never fall back to running unsandboxed
        }
    }
    execvp(argv[0], argv);
    dprintf(STDERR_FILENO, "hx-exec: cannot run %s: %s\n", argv[0], strerror(errno));
    _exit(127);  // same code a shell uses for "command not found"
}

long ms_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    Options o = parse_args(argc, argv);

    // Build argv for execvp BEFORE forking: no allocation in the child.
    std::vector<char*> child_argv;
    for (auto& s : o.command) child_argv.push_back(s.data());
    child_argv.push_back(nullptr);

    // Build the sandbox profile before forking too (it allocates).
    std::string profile = o.sandbox ? hx::build_profile(o.sandbox_opts) : "";

    int fds[2];
    if (pipe(fds) != 0) fail(std::string("pipe: ") + strerror(errno));

    install_signal_forwarding();
    auto started = Clock::now();
    pid_t pid = fork();
    if (pid < 0) fail(std::string("fork: ") + strerror(errno));
    if (pid == 0) {
        close(fds[0]);
        exec_child(o, profile, fds[1], child_argv.data());
    }

    setpgid(pid, pid);  // also set from the parent, to win the race with exec
    close(fds[1]);      // otherwise we'd never see EOF: we'd be holding the write end open ourselves
    int out = fds[0];
    fcntl(out, F_SETFL, O_NONBLOCK);

    hx::OutputCapture capture(o.max_output);
    auto deadline = started + std::chrono::milliseconds(static_cast<long>(o.timeout_s * 1000));
    bool timed_out = false, interrupted = false, child_done = false, pipe_open = true;
    int status = 0;
    Clock::time_point exited_at{};
    char buf[65536];

    while (pipe_open || !child_done) {
        // 1. Read whatever output is available (wait at most 50 ms).
        if (pipe_open) {
            struct pollfd p {out, POLLIN, 0};
            int r = poll(&p, 1, 50);
            if (r > 0) {
                for (;;) {
                    ssize_t n = read(out, buf, sizeof buf);
                    if (n > 0) capture.append(buf, static_cast<size_t>(n));
                    else if (n == 0) { pipe_open = false; break; }  // EOF: every writer closed it
                    else break;                                      // EAGAIN: drained for now
                }
            }
        } else {
            usleep(20000);
        }

        // 2. Has the command exited?
        if (!child_done && waitpid(pid, &status, WNOHANG) == pid) {
            child_done = true;
            exited_at = Clock::now();
        }

        // 3. A background process (`server &`) can keep the pipe open after the command exits.
        //    Give it a moment to flush, then stop reading.
        if (child_done && pipe_open && ms_since(exited_at) > 200) pipe_open = false;

        // 4. Timeout or a forwarded signal: stop the whole group, politely then firmly.
        bool stop = !child_done && (Clock::now() >= deadline || g_forward_signal);
        if (stop) {
            timed_out = !g_forward_signal;
            interrupted = g_forward_signal != 0;
            kill(-pid, SIGTERM);
            auto grace = Clock::now() + std::chrono::seconds(2);
            while (Clock::now() < grace && waitpid(pid, &status, WNOHANG) != pid) usleep(20000);
            kill(-pid, SIGKILL);
            if (waitpid(pid, &status, WNOHANG) != pid) waitpid(pid, &status, 0);
            child_done = true;
            exited_at = Clock::now();
        }
    }
    close(out);
    kill(-pid, SIGKILL);  // clean up anything the command left running in its group (ESRCH if none)

    int exit_code = -1;
    std::string signal_json = "null";
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        int sig = WTERMSIG(status);
        exit_code = 128 + sig;  // shell convention
        signal_json = hx::json_string(strsignal(sig));
    }

    std::cout << "{\"exit_code\":" << exit_code << ",\"signal\":" << signal_json
              << ",\"timed_out\":" << (timed_out ? "true" : "false")
              << ",\"interrupted\":" << (interrupted ? "true" : "false")
              << ",\"sandboxed\":" << (o.sandbox ? "true" : "false")
              << ",\"duration_ms\":" << ms_since(started) << ",\"output_bytes\":" << capture.total_bytes()
              << ",\"truncated_bytes\":" << capture.truncated_bytes() << ",\"output\":" << hx::json_string(capture.text())
              << "}\n";
    return 0;  // the command's own status is in the JSON; nonzero here means hx-exec itself failed
}
