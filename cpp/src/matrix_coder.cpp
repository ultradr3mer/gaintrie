#include "matrix_coder.hpp"

#include <algorithm>
#include <stdexcept>

#include "code_utils.hpp"
#include "pair_writer.hpp"
#include "values.hpp"

namespace cle {
namespace {

constexpr int LEN_BITS = 32;

void put_len(BitWriter& out, uint64_t len, MatrixStats& st) {
    if (len >> LEN_BITS) throw std::runtime_error("segment too long for the 32-bit length field");
    out.put_int(len, LEN_BITS);
    st.framing += LEN_BITS;
}

std::vector<uint64_t> leaf_values(const std::vector<Leaf>& leaves) {
    std::vector<uint64_t> v;
    v.reserve(leaves.size());
    for (const Leaf& l : leaves) v.push_back(l.value);
    return v;
}

// bin codes: Huffman over weights 2^(max - global length)
std::vector<int> bin_lengths(const std::vector<uint64_t>& leaves, const ValueIndex& gvi,
                             const std::vector<int>& global_len, int max_len) {
    std::vector<uint64_t> weights;
    weights.reserve(leaves.size());
    for (uint64_t v : leaves) weights.push_back(1ull << (max_len - global_len[gvi[v]]));
    return huffman_lengths(weights);
}

// stream segment: canonical codes of `lengths` for x[0..n), leaves indexed by `vi`
void put_stream(BitWriter& out, const uint64_t* x, size_t n, const Uniq& u, ValueIndex& vi,
                const std::vector<uint64_t>& leaves, const std::vector<int>& lengths, MatrixStats& st) {
    vi.assign(leaves);
    uint64_t bits = 0;
    for (size_t i = 0; i < u.values.size(); i++) bits += u.counts[i] * lengths[vi[u.values[i]]];
    put_len(out, bits, st);
    std::vector<uint64_t> codes = canonical_codes(lengths);
    for (size_t i = 0; i < n; i++) {
        uint32_t s = vi[x[i]];
        out.put_int(codes[s], lengths[s]);
    }
    st.stream += bits;
}

}  // namespace

std::vector<uint8_t> encode_matrix(const uint64_t* x, size_t n, int width, int mode, uint64_t bin_size,
                                   MatrixStats* stats) {
    if (n == 0) throw std::runtime_error("empty input");
    if (n >> 32) throw std::runtime_error("item count does not fit the 32-bit header field");
    if (bin_size == 0 || bin_size >> 32) throw std::runtime_error("bin size must be 1..2^32-1");
    if (mode != GLOBAL && mode != HYBRID) throw std::runtime_error("unknown matrix mode");
    check_width(x, n, width);
    MatrixStats st;
    BitWriter out;
    out.put_int(n, 32);
    out.put_int(width - 1, 6);
    out.put_int(mode, 1);
    out.put_int(bin_size, 32);
    st.header = 71;

    Uniq u = unique_counts(x, n, width);
    TreeOut seg = best_tree(build_tree(u.values, width));
    ValueIndex vi(width);
    vi.assign(u.values);
    std::vector<uint64_t> cnt;
    for (const Leaf& l : seg.leaves) cnt.push_back(u.counts[vi[l.value]]);
    std::vector<int> lengths = huffman_lengths(cnt);
    st.global_tree = seg.bw.bits();
    put_code_lengths(seg.bw, lengths);
    st.global_lengths = seg.bw.bits() - st.global_tree;
    put_len(out, seg.bw.bits(), st);
    out.append(seg.bw);
    std::vector<uint64_t> leaves = leaf_values(seg.leaves);

    if (mode == GLOBAL) {
        put_stream(out, x, n, u, vi, leaves, lengths, st);
    } else {
        ValueIndex gvi(width);
        gvi.assign(leaves);
        int max_len = *std::max_element(lengths.begin(), lengths.end());
        ValueIndex bvi(width);
        for (size_t start = 0; start < n; start += bin_size) {
            size_t m = std::min<size_t>(bin_size, n - start);
            const uint64_t* b = x + start;
            Uniq bu = unique_counts(b, m, width);
            TreeOut bseg = best_tree(build_tree(bu.values, width));
            put_len(out, bseg.bw.bits(), st);
            out.append(bseg.bw);
            st.bin_trees += bseg.bw.bits();
            std::vector<uint64_t> bleaves = leaf_values(bseg.leaves);
            std::vector<int> blen = bin_lengths(bleaves, gvi, lengths, max_len);
            put_stream(out, b, m, bu, bvi, bleaves, blen, st);
        }
    }
    if (out.bits() != st.total()) throw std::logic_error("size mismatch");
    if (stats) *stats = st;
    return out.bytes();
}

namespace {

void decode_matrix(const uint8_t* data, size_t size, int* width, const std::function<void(uint64_t)>& begin,
                   const SymSink& sink) {
    BitReader br(data, size);
    uint64_t n = br.get_int(32);
    int w = int(br.get_int(6)) + 1;
    int mode = int(br.get_int(1));
    uint64_t bin_size = br.get_int(32);
    if (width) *width = w;
    auto segment = [&]() {
        uint64_t len = br.get_int(LEN_BITS);
        BitReader s = br.sub(len);
        br.skip(len);
        return s;
    };
    BitReader g = segment();
    ReadTree gt = read_pair_tree(g, n);
    if (gt.width != w) throw std::runtime_error("tree width does not match the header");
    std::vector<int> lengths = read_code_lengths(g, gt.leaves.size());
    std::vector<uint64_t> leaves = leaf_values(gt.leaves);
    begin(n);
    if (mode == GLOBAL) {
        BitReader s = segment();
        decode_chunks(s, PrefixDecoder(canonical_codes(lengths), lengths), 0, n, leaves, sink);
        return;
    }
    if (bin_size == 0) throw std::runtime_error("bin size 0");
    ValueIndex gvi(w);
    gvi.assign(leaves);
    int max_len = *std::max_element(lengths.begin(), lengths.end());
    for (uint64_t start = 0; start < n; start += bin_size) {
        uint64_t m = std::min(bin_size, n - start);
        BitReader t = segment();
        ReadTree bt = read_pair_tree(t, m);
        std::vector<uint64_t> bleaves = leaf_values(bt.leaves);
        std::vector<int> blen = bin_lengths(bleaves, gvi, lengths, max_len);
        BitReader s = segment();
        decode_chunks(s, PrefixDecoder(canonical_codes(blen), blen), start, m, bleaves, sink);
    }
}

}  // namespace

std::vector<uint64_t> decode_matrix(const uint8_t* data, size_t size, int* width) {
    std::vector<uint64_t> out;
    decode_matrix(data, size, width, [&](uint64_t n) { out.resize(n); },
                  [&](uint64_t start, const uint32_t* sym, size_t m, const std::vector<uint64_t>& leaves) {
                      for (size_t i = 0; i < m; i++) out[start + i] = leaves[sym[i]];
                  });
    return out;
}

std::string decode_matrix_le(const uint8_t* data, size_t size, int bytes) {
    std::string out;
    decode_matrix(data, size, nullptr, [&](uint64_t n) { out.resize(n * bytes); },
                  [&](uint64_t start, const uint32_t* sym, size_t m, const std::vector<uint64_t>& leaves) {
                      put_le(leaves, sym, m, bytes, &out[start * bytes]);
                  });
    return out;
}

}  // namespace cle
