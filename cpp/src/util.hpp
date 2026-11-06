#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cle {

inline std::string base64_encode(const std::string& in) {
    static const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t v = uint32_t(uint8_t(in[i])) << 16 | uint32_t(uint8_t(in[i + 1])) << 8 | uint8_t(in[i + 2]);
        out += A[v >> 18], out += A[(v >> 12) & 63], out += A[(v >> 6) & 63], out += A[v & 63];
    }
    if (size_t r = in.size() - i) {
        uint32_t v = uint32_t(uint8_t(in[i])) << 16 | (r == 2 ? uint32_t(uint8_t(in[i + 1])) << 8 : 0);
        out += A[v >> 18], out += A[(v >> 12) & 63];
        out += r == 2 ? A[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

inline std::string base64_decode(const std::string& in) {
    std::string out;
    uint32_t acc = 0;
    int n = 0;
    for (char c : in) {
        int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26
              : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' ? 62 : c == '/' ? 63 : -1;
        if (v < 0) {
            if (c == '=') break;
            throw std::runtime_error("invalid base64");
        }
        acc = acc << 6 | uint32_t(v);
        if ((n += 6) >= 8) {
            n -= 8;
            out += char((acc >> n) & 0xFF);
        }
    }
    return out;
}

// fixed workers; wait() rethrows the first task exception
class ThreadPool {
public:
    explicit ThreadPool(int n) {
        for (int i = 0; i < std::max(1, n); i++) workers_.emplace_back([this] { run(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> l(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> l(m_);
            q_.push_back(std::move(f));
            busy_++;
        }
        cv_.notify_one();
    }
    void wait() {
        std::unique_lock<std::mutex> l(m_);
        done_.wait(l, [this] { return busy_ == 0; });
        if (err_) std::rethrow_exception(std::exchange(err_, nullptr));
    }

private:
    void run() {
        while (true) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [this] { return stop_ || !q_.empty(); });
                if (q_.empty()) return;
                f = std::move(q_.front());
                q_.pop_front();
            }
            try {
                if (!failed_) f();
            } catch (...) {
                std::lock_guard<std::mutex> l(m_);
                if (!err_) err_ = std::current_exception();
                failed_ = true;
            }
            std::lock_guard<std::mutex> l(m_);
            if (--busy_ == 0) done_.notify_all();
        }
    }
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> q_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    size_t busy_ = 0;
    bool stop_ = false;
    std::atomic<bool> failed_{false};
    std::exception_ptr err_;
};

}  // namespace cle
