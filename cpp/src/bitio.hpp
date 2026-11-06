#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cle {

// MSB-first bit packing, same layout as clarautils BitWriter / BitReader
class BitWriter {
public:
    void put(bool b) { put_small(b ? 1 : 0, 1); }

    void put_int(uint64_t v, int len) {
        if (len <= 0) return;
        if (len > 32) {
            put_small((v >> 32) & mask(len - 32), len - 32);
            put_small(v & 0xffffffffull, 32);
        } else {
            put_small(v & mask(len), len);
        }
    }

    void append(const BitWriter& o) {
        if (nacc_ == 0) {
            bytes_.insert(bytes_.end(), o.bytes_.begin(), o.bytes_.end());
        } else {
            for (uint8_t b : o.bytes_) put_small(b, 8);
        }
        put_small(o.acc_, o.nacc_);
    }

    uint64_t bits() const { return bytes_.size() * 8 + nacc_; }

    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> out(bytes_);
        if (nacc_) out.push_back(uint8_t(acc_ << (8 - nacc_)));
        return out;
    }

private:
    static uint64_t mask(int n) { return n >= 64 ? ~0ull : (1ull << n) - 1; }

    void put_small(uint64_t v, int n) {
        if (n == 0) return;
        acc_ = (acc_ << n) | v;
        nacc_ += n;
        while (nacc_ >= 8) {
            nacc_ -= 8;
            bytes_.push_back(uint8_t(acc_ >> nacc_));
        }
        acc_ &= (1ull << nacc_) - 1;
    }

    std::vector<uint8_t> bytes_;
    uint64_t acc_ = 0;
    int nacc_ = 0;
};

class BitReader {
public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size), pos_(0), end_(uint64_t(size) * 8) {}

    BitReader sub(uint64_t len) const {
        need(len);
        BitReader r = *this;
        r.end_ = pos_ + len;
        return r;
    }

    bool get_bool() {
        need(1);
        bool b = (data_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1;
        ++pos_;
        return b;
    }

    uint64_t get_int(int n) {
        if (n <= 0) return 0;
        if (n > 56) {
            uint64_t hi = get_int(n - 32);
            return (hi << 32) | get_int(32);
        }
        need(n);
        uint64_t v = peek(n);
        pos_ += n;
        return v;
    }

    // n <= 56 bits at the current position, zero padded past the data
    uint64_t peek(int n) const {
        if (n == 0) return 0;
        uint64_t byte = pos_ >> 3, w = 0;
        if (byte + 8 <= size_) {
            std::memcpy(&w, data_ + byte, 8);
            w = __builtin_bswap64(w);
        } else {
            for (int i = 0; i < 8; i++) w = (w << 8) | (byte + i < size_ ? data_[byte + i] : 0);
        }
        return (w << (pos_ & 7)) >> (64 - n);
    }

    void skip(uint64_t n) {
        need(n);
        pos_ += n;
    }

    uint64_t pos() const { return pos_; }
    uint64_t left() const { return end_ - pos_; }

private:
    void need(uint64_t n) const {
        if (n > end_ - pos_) throw std::runtime_error("read past end of data");
    }

    const uint8_t* data_;
    size_t size_;
    uint64_t pos_, end_;
};

}  // namespace cle
