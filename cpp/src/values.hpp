#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cle {

struct Uniq {
    std::vector<uint64_t> values, counts;
};

// distinct values (sorted) + counts; dense counting for narrow widths
inline Uniq unique_counts(const uint64_t* x, size_t n, int width) {
    Uniq u;
    if (width <= 20 && (n << 4) >= (size_t(1) << width)) {
        std::vector<uint32_t> cnt(size_t(1) << width, 0);
        for (size_t i = 0; i < n; i++) cnt[x[i]]++;
        for (size_t v = 0; v < cnt.size(); v++)
            if (cnt[v]) {
                u.values.push_back(v);
                u.counts.push_back(cnt[v]);
            }
        return u;
    }
    std::vector<uint64_t> s(x, x + n);
    std::sort(s.begin(), s.end());
    for (size_t i = 0; i < n;) {
        size_t j = i;
        while (j < n && s[j] == s[i]) j++;
        u.values.push_back(s[i]);
        u.counts.push_back(j - i);
        i = j;
    }
    return u;
}

inline void check_width(const uint64_t* x, size_t n, int width) {
    if (width < 1 || width > 64) throw std::runtime_error("width must be 1..64");
    if (width == 64) return;
    uint64_t any = 0;
    for (size_t i = 0; i < n; i++) any |= x[i];
    if (any >> width) throw std::runtime_error("value does not fit width");
}

// value -> index; reusable: assign() only overwrites the given values
class ValueIndex {
public:
    explicit ValueIndex(int width) : dense_(width <= 20) {
        if (dense_) table_.resize(size_t(1) << width);
    }

    void assign(const std::vector<uint64_t>& values) {
        if (!dense_) map_.clear();
        for (size_t i = 0; i < values.size(); i++) {
            if (dense_)
                table_[values[i]] = uint32_t(i);
            else
                map_[values[i]] = uint32_t(i);
        }
    }

    uint32_t operator[](uint64_t v) const { return dense_ ? table_[v] : map_.at(v); }

private:
    bool dense_;
    std::vector<uint32_t> table_;
    std::unordered_map<uint64_t, uint32_t> map_;
};

}  // namespace cle
