#pragma once

#include <algorithm>
#include <string>

namespace hx {

// Collects a command's output within a byte budget, keeping the HEAD and the TAIL.
//
// Why both ends? The start of output shows what ran (and the first error); the end shows how it finished
// (the final error, the test summary, the exit message). The middle of a 50,000-line build log is rarely
// what anyone needs. Memory stays bounded however much the command prints.
class OutputCapture {
public:
    explicit OutputCapture(size_t limit) : head_limit_(limit / 2), tail_limit_(limit - limit / 2) {}

    void append(const char* data, size_t n) {
        total_ += n;
        size_t take = std::min(n, head_limit_ - head_.size());
        head_.append(data, take);
        data += take;
        n -= take;
        if (n == 0) return;
        tail_.append(data, n);
        // Trim occasionally (not on every append) to keep this amortized O(n).
        if (tail_.size() > 2 * tail_limit_ + 4096) tail_.erase(0, tail_.size() - tail_limit_);
    }

    size_t total_bytes() const { return total_; }

    size_t truncated_bytes() const {
        size_t kept = head_.size() + std::min(tail_.size(), tail_limit_);
        return total_ > kept ? total_ - kept : 0;
    }

    std::string text() const {
        if (truncated_bytes() == 0) return head_ + tail_;
        std::string tail = tail_.substr(tail_.size() - tail_limit_);
        return head_ + "\n\n... [" + std::to_string(truncated_bytes()) + " bytes of output truncated] ...\n\n" + tail;
    }

private:
    size_t head_limit_;
    size_t tail_limit_;
    size_t total_ = 0;
    std::string head_;
    std::string tail_;
};

}  // namespace hx
