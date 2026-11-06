#include "gain_coder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cle {
namespace {

// float32 like clarautils get_bitwise_entropy; numpy's float32 log2 is not
// correctly rounded, so rare ties can still break differently
float l2(float x) { return static_cast<float>(std::log2(static_cast<double>(x))); }

float entropy(uint64_t k, uint64_t n) {
    float p = static_cast<float>(static_cast<double>(k) / static_cast<double>(n));
    if (!(p > 0.0f && p < 1.0f)) return 0.0f;
    float q = 1.0f - p;
    return (-p) * l2(p) - q * l2(q);
}

// numpy float32 add.reduce (pairwise sum, n < 128)
float np_sum(const float* a, int n) {
    float res;
    if (n < 8) {
        res = -0.0f;
        for (int i = 0; i < n; i++) res += a[i];
    } else {
        float r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        int i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
    }
    return 0.0f + res;
}

struct Builder {
    int width;
    std::vector<TNode>& nodes;
    std::vector<uint32_t> both;  // [i * 64 + j], i < j: items with bit i and bit j

    int bit(uint64_t v, int p) const { return int((v >> (width - 1 - p)) & 1); }

    int add(const TNode& t) {
        nodes.push_back(t);
        return int(nodes.size()) - 1;
    }

    // find_split: LSB index i = 0..w-1, strict > from 0
    int find_split(const uint64_t* it, size_t n, uint64_t rem) {
        int pos[64], w = 0;
        for (uint64_t r = rem; r; r &= r - 1) pos[w++] = __builtin_ctzll(r);
        uint64_t ones[64] = {};
        for (int i = 0; i < w; i++) std::fill_n(both.begin() + i * 64, w, 0u);
        for (size_t k = 0; k < n; k++) {
            uint64_t v = it[k], m = 0;
            for (int j = 0; j < w; j++) m |= uint64_t(bit(v, pos[j])) << j;
            for (uint64_t a = m; a; a &= a - 1) {
                int i = __builtin_ctzll(a);
                ones[i]++;
                for (uint64_t b = a & (a - 1); b; b &= b - 1) both[i * 64 + __builtin_ctzll(b)]++;
            }
        }
        float begin_sum = 0.0f;
        for (int i = 0; i < w; i++) begin_sum = begin_sum + entropy(ones[w - 1 - i], n);
        float max_g = 0.0f, e1[64], e0[64];
        int best = 0;
        for (int i = 0; i < w; i++) {
            int m = w - 1 - i;
            uint64_t n1 = ones[m], n0 = n - n1;
            int c = 0;
            for (int j = w - 1; j >= 0; j--) {
                if (j == m) continue;
                uint64_t k1 = j < m ? both[j * 64 + m] : both[m * 64 + j];
                e1[c] = entropy(k1, n1);
                e0[c] = entropy(ones[j] - k1, n0);
                c++;
            }
            float g = ((begin_sum - np_sum(e1, c)) * float(n1) + (begin_sum - np_sum(e0, c)) * float(n0)) /
                      float(n1 + n0);
            if (g > max_g) {
                max_g = g;
                best = i;
            }
        }
        return pos[w - 1 - best];
    }

    int build(uint64_t* it, size_t n, uint64_t rem) {
        if (n == 1) {
            TNode t;
            t.kind = LEAF;
            t.value = it[0];
            return add(t);
        }
        uint64_t all = ~0ull, any = 0;
        for (size_t k = 0; k < n; k++) {
            all &= it[k];
            any |= it[k];
        }
        int chain[64], nchain = 0;
        for (uint64_t r = rem; r; r &= r - 1) {
            int p = __builtin_ctzll(r);
            if (bit(all, p) == bit(any, p)) {
                TNode t;
                t.kind = STRAIT;
                t.pos = uint8_t(p);
                t.bit = uint8_t(bit(all, p));
                chain[nchain++] = add(t);
                rem &= ~(1ull << p);
            }
        }
        int s = find_split(it, n, rem);
        uint64_t crem = rem & ~(1ull << s);
        uint64_t* mid = std::partition(it, it + n, [&](uint64_t v) { return bit(v, s) == 1; });
        TNode t;
        t.kind = SPLIT;
        t.pos = uint8_t(s);
        int idx = add(t);
        int ta = build(it, size_t(mid - it), crem);
        int fb = build(mid, size_t(it + n - mid), crem);
        nodes[idx].a = ta;
        nodes[idx].b = fb;
        nodes[idx].leaves = nodes[ta].leaves + nodes[fb].leaves;
        for (int c = 0; c < nchain; c++) {
            nodes[chain[c]].a = c + 1 < nchain ? chain[c + 1] : idx;
            nodes[chain[c]].leaves = nodes[idx].leaves;
        }
        return nchain ? chain[0] : idx;
    }
};

}  // namespace

Tree build_tree(std::vector<uint64_t> values, int width) {
    if (width < 1 || width > 64) throw std::runtime_error("width must be 1..64");
    if (values.empty()) throw std::runtime_error("no values");
    std::sort(values.begin(), values.end());
    if (std::adjacent_find(values.begin(), values.end()) != values.end())
        throw std::runtime_error("values must be distinct");
    if (width < 64 && values.back() >> width) throw std::runtime_error("value does not fit width");
    Tree tree;
    tree.width = width;
    tree.nodes.reserve(values.size() * 3);
    Builder b{width, tree.nodes, std::vector<uint32_t>(64 * 64)};
    tree.root = b.build(values.data(), values.size(), full_rem(width));
    return tree;
}

}  // namespace cle
