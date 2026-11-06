#include "code_utils.hpp"

#include <algorithm>
#include <climits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace cle {

int tb_bits(uint64_t idx, uint64_t n) {
    if (n <= 1) return 0;
    int k = pos_bits(n);
    return idx < (1ull << k) - n ? k - 1 : k;
}

void put_tb(BitWriter& bw, uint64_t idx, uint64_t n) {
    if (n <= 1) return;
    int k = pos_bits(n);
    uint64_t u = (1ull << k) - n;
    if (idx < u)
        bw.put_int(idx, k - 1);
    else
        bw.put_int(idx + u, k);
}

uint64_t read_tb(BitReader& br, uint64_t n) {
    if (n <= 1) return 0;
    int k = pos_bits(n);
    uint64_t u = (1ull << k) - n;
    uint64_t x = br.get_int(k - 1);
    return x < u ? x : ((x << 1) | uint64_t(br.get_bool())) - u;
}

namespace {
struct CombTable {
    uint64_t c[65][65] = {};
    CombTable() {
        for (int n = 0; n <= 64; n++) {
            c[n][0] = 1;
            for (int k = 1; k <= n; k++) c[n][k] = c[n - 1][k - 1] + (k <= n - 1 ? c[n - 1][k] : 0);
        }
    }
};
const CombTable COMB;
}  // namespace

uint64_t comb(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0;
    if (n > 64) throw std::runtime_error("comb out of range");
    return COMB.c[n][k];
}

uint64_t colex_rank(const std::vector<int>& idx) {
    uint64_t r = 0;
    for (size_t k = 0; k < idx.size(); k++) r += comb(idx[k], int(k) + 1);
    return r;
}

std::vector<int> colex_unrank(uint64_t rank, int count) {
    std::vector<int> out(count);
    for (int k = count; k > 0; k--) {
        int c = k - 1;
        while (c < 64 && comb(c + 1, k) <= rank) c++;
        rank -= comb(c, k);
        out[k - 1] = c;
    }
    return out;
}

// heapq order of the Python reference: (count, insertion counter)
std::vector<int> huffman_lengths(const std::vector<uint64_t>& counts) {
    size_t n = counts.size();
    if (n == 1) return {0};
    using Item = std::tuple<uint64_t, uint64_t, uint32_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    std::vector<uint32_t> parent(2 * n, 0);
    for (size_t i = 0; i < n; i++) heap.emplace(counts[i], i, uint32_t(i));
    uint64_t tie = n;
    uint32_t next = uint32_t(n);
    while (heap.size() > 1) {
        auto [c1, t1, s1] = heap.top();
        heap.pop();
        auto [c2, t2, s2] = heap.top();
        heap.pop();
        parent[s1] = parent[s2] = next;
        heap.emplace(c1 + c2, tie++, next++);
    }
    std::vector<int> depth(next, 0);
    for (uint32_t i = next - 1; i-- > 0;) depth[i] = depth[parent[i]] + 1;
    return std::vector<int>(depth.begin(), depth.begin() + n);
}

std::vector<uint64_t> canonical_codes(const std::vector<int>& lengths) {
    std::vector<uint32_t> order(lengths.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return lengths[a] < lengths[b]; });
    std::vector<uint64_t> codes(lengths.size());
    uint64_t code = 0;
    int prev = 0;
    for (uint32_t i : order) {
        code <<= lengths[i] - prev;
        prev = lengths[i];
        codes[i] = prev ? code : 0;
        code++;
    }
    return codes;
}

uint64_t code_length_bits(const std::vector<int>& lengths) {
    int max_len = *std::max_element(lengths.begin(), lengths.end());
    uint64_t bits = MAX_LEN_BITS;
    if (max_len)
        for (int l : lengths) bits += tb_bits(l - 1, max_len);
    return bits;
}

void put_code_lengths(BitWriter& bw, const std::vector<int>& lengths) {
    int max_len = *std::max_element(lengths.begin(), lengths.end());
    if (max_len >= 1 << MAX_LEN_BITS) throw std::runtime_error("huffman code too long");
    bw.put_int(max_len, MAX_LEN_BITS);
    if (max_len)
        for (int l : lengths) put_tb(bw, l - 1, max_len);
}

std::vector<int> read_code_lengths(BitReader& br, size_t count) {
    int max_len = int(br.get_int(MAX_LEN_BITS));
    std::vector<int> out(count, 0);
    if (max_len)
        for (auto& l : out) l = int(read_tb(br, max_len)) + 1;
    return out;
}

PrefixDecoder::PrefixDecoder(const std::vector<uint64_t>& codes, const std::vector<int>& lens) {
    if (codes.empty()) throw std::runtime_error("empty code table");
    if (codes.size() == 1 && lens[0] == 0) {
        single_ = true;
        return;
    }
    child_.assign(2, 0);
    int max_len = 0;
    for (size_t s = 0; s < codes.size(); s++) {
        int len = lens[s];
        if (len <= 0 || len > 64) throw std::runtime_error("invalid code length");
        max_len = std::max(max_len, len);
        int32_t node = 0;
        for (int d = 0; d < len; d++) {
            int bit = int((codes[s] >> (len - 1 - d)) & 1);
            int32_t& c = child_[2 * node + bit];
            if (c < 0) throw std::runtime_error("not a prefix code");
            if (d == len - 1) {
                if (c != 0) throw std::runtime_error("not a prefix code");
                c = -int32_t(s) - 1;
            } else {
                if (c == 0) {
                    c = int32_t(child_.size() / 2);
                    child_.resize(child_.size() + 2, 0);
                }
                node = child_[2 * node + bit];
            }
        }
    }
    k_ = std::min(max_len, 14);
    table_.resize(size_t(1) << k_);
    for (uint32_t pat = 0; pat < table_.size(); pat++) {
        int32_t node = 0;
        Entry e{INT32_MIN, 0};
        for (int d = 0; d < k_; d++) {
            int32_t c = child_[2 * node + ((pat >> (k_ - 1 - d)) & 1)];
            if (c < 0) {
                e = {-c - 1, uint8_t(d + 1)};
                break;
            }
            if (c == 0) break;
            node = c;
            if (d == k_ - 1) e = {-node, uint8_t(k_)};
        }
        table_[pat] = e;
    }
}

void PrefixDecoder::decode(BitReader& br, size_t n, uint32_t* out) const {
    if (single_) {
        std::fill(out, out + n, 0u);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        Entry e = table_[br.peek(k_)];
        if (e.v == INT32_MIN) throw std::runtime_error("invalid code in stream");
        br.skip(e.len);
        int32_t v = e.v;
        while (v < 0) {
            int32_t c = child_[2 * (-v) + br.get_bool()];
            if (c == 0) throw std::runtime_error("invalid code in stream");
            v = c < 0 ? -c - 1 : -c;
        }
        out[i] = uint32_t(v);
    }
}

void decode_chunks(BitReader& br, const PrefixDecoder& dec, uint64_t start, uint64_t n,
                   const std::vector<uint64_t>& leaves, const SymSink& sink) {
    const uint64_t chunk = 1 << 16;
    std::vector<uint32_t> sym(std::min(n, chunk));
    for (uint64_t i = 0; i < n; i += chunk) {
        size_t k = size_t(std::min(chunk, n - i));
        dec.decode(br, k, sym.data());
        sink(start + i, sym.data(), k, leaves);
    }
}

}  // namespace cle
