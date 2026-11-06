#include "pair_writer.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "code_utils.hpp"
#include "values.hpp"

namespace cle {
namespace {

uint64_t rev64(uint64_t v) {
    v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
    v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
    return __builtin_bswap64(v);
}

int index_in(uint64_t rem, int p) { return popcount(rem & ((1ull << p) - 1)); }

int permutation_bits(const std::vector<int>& order) {
    std::vector<int> left(order.size());
    std::iota(left.begin(), left.end(), 0);
    int bits = 0;
    for (size_t i = 0; i < order.size(); i++) {
        auto it = std::find(left.begin(), left.end(), order[i]);
        bits += tb_bits(uint64_t(it - left.begin()), order.size() - i);
        left.erase(it);
    }
    return bits;
}

}  // namespace

bool PairWriter::Codes::operator==(const Codes& o) const {
    for (int k = 0; k < 4; k++)
        if (bits[k] != o.bits[k] || len[k] != o.len[k]) return false;
    return true;
}

// 0 / 10 / 110 / ... by frequency rank, the last one without the 0
PairWriter::Codes PairWriter::kind_codes(const std::vector<int>& order) {
    Codes c;
    int n = int(order.size());
    for (int i = 0; i < n; i++) {
        uint64_t ones = (1ull << i) - 1;
        c.bits[order[i]] = i < n - 1 ? ones << 1 : ones;
        c.len[order[i]] = i < n - 1 ? i + 1 : i;
    }
    return c;
}

// Counter.most_common: count desc, ties by first occurrence
std::vector<int> PairWriter::most_common(const std::vector<Entry>& nodes) {
    uint64_t count[4] = {};
    size_t first[4] = {~size_t(0), ~size_t(0), ~size_t(0), ~size_t(0)};
    for (size_t i = 0; i < nodes.size(); i++) {
        int k = nodes[i].kind;
        if (!count[k]++) first[k] = i;
    }
    std::vector<int> order;
    for (int k = 0; k < 4; k++)
        if (count[k]) order.push_back(k);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return count[a] != count[b] ? count[a] > count[b] : first[a] < first[b];
    });
    return order;
}

PairWriter::PairWriter(const Tree& tree, int mask_mode)
    : tree_(tree), width_(tree.width), mask_mode_(mask_mode), pair_(tree.nodes.size(), 0) {
    choose();
}

uint64_t PairWriter::pmask(uint64_t v) const { return rev64(v) >> (64 - width_); }

void PairWriter::two_leaves(int node, uint64_t& a, uint64_t& b) const {
    const auto& n = tree_.nodes;
    while (n[node].kind == STRAIT) node = n[node].a;
    a = n[n[node].a].value;
    b = n[n[node].b].value;
}

std::vector<int> PairWriter::pair_diff(uint64_t a, uint64_t b, uint64_t rem) const {
    std::vector<int> diff;
    for (uint64_t d = rem & pmask(a ^ b); d; d &= d - 1) diff.push_back(index_in(rem, __builtin_ctzll(d)));
    return diff;
}

uint64_t PairWriter::mask_bits(int r, const std::vector<int>& diff) const {
    int d = int(diff.size());
    return mask_mode_ == RAW ? uint64_t(r) : tb_bits(d - 1, r) + tb_bits(colex_rank(diff), comb(r, d));
}

uint64_t PairWriter::pair_cost(int node, uint64_t rem, const Codes& codes) const {
    uint64_t a, b;
    two_leaves(node, a, b);
    int r = popcount(rem);
    return codes.len[PAIR] + mask_bits(r, pair_diff(a, b, rem)) + r - 1;
}

uint64_t PairWriter::classic_cost(int node, uint64_t rem, const Codes& codes) const {
    const TNode& t = tree_.nodes[node];
    int r = popcount(rem);
    uint64_t tb = rem ? codes.len[t.kind] : 0;
    if (t.kind == LEAF) return tb + r;
    uint64_t crem = rem & ~(1ull << t.pos);
    uint64_t own = tb + tb_bits(index_in(rem, t.pos), r);
    if (t.kind == STRAIT) return own + 1 + classic_cost(t.a, crem, codes);
    return own + classic_cost(t.a, crem, codes) + classic_cost(t.b, crem, codes);
}

void PairWriter::heads(int node, uint64_t rem, std::vector<std::pair<int, uint64_t>>& out) const {
    const TNode& t = tree_.nodes[node];
    if (t.kind == LEAF) return;
    if (t.leaves == 2) {
        out.emplace_back(node, rem);
        return;
    }
    uint64_t crem = rem & ~(1ull << t.pos);
    heads(t.a, crem, out);
    if (t.kind == SPLIT) heads(t.b, crem, out);
}

void PairWriter::choose() {
    std::vector<std::pair<int, uint64_t>> hs;
    heads(tree_.root, full_rem(width_), hs);
    Codes codes = kind_codes({STRAIT, LEAF, SPLIT, PAIR});
    for (int iter = 0; iter < 4; iter++) {  // type code lengths depend on the choices
        std::fill(pair_.begin(), pair_.end(), 0);
        for (auto [head, rem] : hs) {
            int best = -1;
            uint64_t best_cost = classic_cost(head, rem, codes), before = 0;
            int t = head;
            uint64_t r = rem;
            while (true) {
                uint64_t cost = before + pair_cost(t, r, codes);
                if (cost < best_cost) {
                    best = t;
                    best_cost = cost;
                }
                const TNode& n = tree_.nodes[t];
                if (n.kind != STRAIT) break;
                before += codes.len[STRAIT] + tb_bits(index_in(r, n.pos), popcount(r)) + 1;
                r &= ~(1ull << n.pos);
                t = n.a;
            }
            if (best >= 0) pair_[best] = 1;
        }
        std::vector<int> order = most_common(walk());
        for (int k = 0; k < 4; k++)
            if (std::find(order.begin(), order.end(), k) == order.end()) order.push_back(k);
        Codes next = kind_codes(order);
        if (next == codes) break;
        codes = next;
    }
}

std::vector<PairWriter::Entry> PairWriter::walk() const {
    std::vector<Entry> out;
    out.push_back({tree_.root, full_rem(width_), 0, 0, 0});
    for (size_t i = 0; i < out.size(); i++) {
        Entry e = out[i];
        const TNode& t = tree_.nodes[e.node];
        e.kind = pair_[e.node] ? PAIR : t.kind;
        out[i].kind = e.kind;
        if (e.kind == PAIR || e.kind == LEAF) continue;
        uint64_t crem = e.rem & ~(1ull << t.pos);
        if (e.kind == STRAIT) {
            out.push_back({t.a, crem, e.code, e.len, 0});
        } else {
            out.push_back({t.a, crem, (e.code << 1) | 1, e.len + 1, 0});
            out.push_back({t.b, crem, e.code << 1, e.len + 1, 0});
        }
    }
    return out;
}

// one order of the positions minimizing the sum of the split / strait indices
std::vector<int> PairWriter::position_order(const std::vector<Entry>& nodes) const {
    int w = width_;
    std::vector<int64_t> W(size_t(w) * w, 0);
    std::vector<double> t_sum(w, 0.0), t_cnt(w, 0.0);
    for (const Entry& e : nodes) {
        if (e.kind != SPLIT && e.kind != STRAIT) continue;
        int p = tree_.nodes[e.node].pos;
        for (uint64_t r = e.rem; r; r &= r - 1) W[size_t(__builtin_ctzll(r)) * w + p] += 1;
        W[size_t(p) * w + p] -= 1;
        t_sum[p] += w - popcount(e.rem);
        t_cnt[p] += 1;
    }
    std::vector<double> avg(w);
    for (int p = 0; p < w; p++) avg[p] = t_cnt[p] > 0 ? t_sum[p] / std::max(t_cnt[p], 1.0) : double(w);
    std::vector<int> order(w);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return avg[a] < avg[b]; });
    std::vector<int64_t> before(w + 1), after(w + 1);
    bool improved = true;
    while (improved) {
        improved = false;
        std::vector<int> snapshot = order;
        for (int p : snapshot) {
            std::vector<int> rest;
            for (int q : order)
                if (q != p) rest.push_back(q);
            size_t m = rest.size();
            before[0] = 0;
            for (size_t s = 0; s < m; s++) before[s + 1] = before[s] + W[size_t(rest[s]) * w + p];
            after[m] = 0;
            for (size_t s = m; s-- > 0;) after[s] = after[s + 1] + W[size_t(p) * w + rest[s]];
            size_t best = 0;
            for (size_t s = 1; s <= m; s++)
                if (before[s] + after[s] < before[best] + after[best]) best = s;
            size_t cur = size_t(std::find(order.begin(), order.end(), p) - order.begin());
            if (before[best] + after[best] < before[cur] + after[cur]) {
                rest.insert(rest.begin() + best, p);
                order = rest;
                improved = true;
            }
        }
    }
    return order;
}

TreeOut PairWriter::write_tree(int pos_mode) const {
    int w = width_;
    std::vector<Entry> nodes = walk();
    std::vector<int> order = most_common(nodes);
    Codes codes = kind_codes(order);

    std::vector<int> pos_order = position_order(nodes);
    std::vector<uint64_t> less(w, 0);
    for (int i = 0; i < w; i++)
        for (int j = 0; j < i; j++) less[pos_order[i]] |= 1ull << pos_order[j];
    uint64_t msb_cost = 0, ord_cost = permutation_bits(pos_order);
    for (const Entry& e : nodes) {
        if (e.kind != SPLIT && e.kind != STRAIT) continue;
        int p = tree_.nodes[e.node].pos, r = popcount(e.rem);
        msb_cost += tb_bits(index_in(e.rem, p), r);
        ord_cost += tb_bits(popcount(e.rem & less[p]), r);
    }
    if (pos_mode == POS_AUTO) pos_mode = ord_cost < msb_cost ? POS_ORDER : POS_MSB;

    TreeOut out;
    out.width = w;
    BitWriter& bw = out.bw;
    bw.put_int(w - 1, 6);
    bw.put_int(mask_mode_, 1);
    bw.put_int(pos_mode, 1);
    bw.put_int(order.size() - 1, 2);
    for (int k : order) bw.put_int(k, 2);
    if (pos_mode == POS_ORDER) {
        std::vector<int> left(w);
        std::iota(left.begin(), left.end(), 0);
        for (int i = 0; i < w; i++) {
            auto it = std::find(left.begin(), left.end(), pos_order[i]);
            put_tb(bw, uint64_t(it - left.begin()), w - i);
            left.erase(it);
        }
    }
    for (const Entry& e : nodes) {
        const TNode& t = tree_.nodes[e.node];
        int r = popcount(e.rem);
        if (e.rem)
            bw.put_int(codes.bits[e.kind], codes.len[e.kind]);
        else if (e.kind != LEAF)
            throw std::logic_error("non-leaf without open positions");
        if (e.kind == LEAF) {
            for (uint64_t m = e.rem; m; m &= m - 1) bw.put(bit(t.value, __builtin_ctzll(m)));
            out.leaves.push_back({t.value, e.code, e.len});
        } else if (e.kind == PAIR) {
            uint64_t a, b;
            two_leaves(e.node, a, b);
            uint64_t dpos = e.rem & pmask(a ^ b);
            int p0 = __builtin_ctzll(dpos);
            uint64_t one = bit(a, p0) ? a : b;
            if (mask_mode_ == RAW) {
                for (uint64_t m = e.rem; m; m &= m - 1) bw.put((dpos >> __builtin_ctzll(m)) & 1);
            } else {
                std::vector<int> diff = pair_diff(a, b, e.rem);
                int d = int(diff.size());
                put_tb(bw, d - 1, r);
                put_tb(bw, colex_rank(diff), comb(r, d));
            }
            for (uint64_t m = e.rem & ~(1ull << p0); m; m &= m - 1) bw.put(bit(one, __builtin_ctzll(m)));
            out.leaves.push_back({one, (e.code << 1) | 1, e.len + 1});
            out.leaves.push_back({a ^ b ^ one, e.code << 1, e.len + 1});
        } else {
            int p = t.pos;
            put_tb(bw, pos_mode == POS_MSB ? index_in(e.rem, p) : popcount(e.rem & less[p]), r);
            if (e.kind == STRAIT) bw.put(t.bit);
        }
    }
    return out;
}

TreeOut write_pair_tree(const Tree& tree, int mask_mode, int pos_mode) {
    return PairWriter(tree, mask_mode).write_tree(pos_mode);
}

TreeOut best_tree(const Tree& tree, int pos_mode) {
    TreeOut raw = write_pair_tree(tree, RAW, pos_mode);
    TreeOut cmb = write_pair_tree(tree, COMB, pos_mode);
    return cmb.bw.bits() < raw.bw.bits() ? cmb : raw;
}

ReadTree read_pair_tree(BitReader& br, uint64_t max_leaves) {
    struct Q {
        uint64_t rem, value, code;
        int len;
    };
    int w = int(br.get_int(6)) + 1;
    int mask_mode = int(br.get_int(1));
    int pos_mode = int(br.get_int(1));
    int nk = int(br.get_int(2)) + 1;
    int order[4];
    for (int i = 0; i < nk; i++) order[i] = int(br.get_int(2));
    std::vector<int> pos_order;
    if (pos_mode == POS_ORDER) {
        std::vector<int> left(w);
        std::iota(left.begin(), left.end(), 0);
        for (int i = 0; i < w; i++) {
            uint64_t idx = read_tb(br, w - i);
            pos_order.push_back(left[idx]);
            left.erase(left.begin() + idx);
        }
    }
    auto shift = [&](int p) { return uint64_t(w - 1 - p); };
    ReadTree out{w, {}};
    std::vector<Q> queue{{full_rem(w), 0, 0, 0}};
    for (size_t qi = 0; qi < queue.size(); qi++) {
        Q q = queue[qi];
        if (out.leaves.size() + (queue.size() - qi) > max_leaves) throw std::runtime_error("tree too large");
        int k = LEAF;
        if (q.rem) {
            int i = 0;
            while (i < nk - 1 && br.get_bool()) i++;
            k = order[i];
        }
        int r = popcount(q.rem);
        int rpos[64], ri = 0;
        for (uint64_t m = q.rem; m; m &= m - 1) rpos[ri++] = __builtin_ctzll(m);
        if (k == LEAF) {
            for (int i = 0; i < r; i++) q.value |= uint64_t(br.get_bool()) << shift(rpos[i]);
            out.leaves.push_back({q.value, q.code, q.len});
            continue;
        }
        if (k == PAIR) {
            std::vector<int> diff;
            if (mask_mode == RAW) {
                for (int i = 0; i < r; i++)
                    if (br.get_bool()) diff.push_back(i);
                if (diff.empty()) throw std::runtime_error("empty pair mask");
            } else {
                int d = int(read_tb(br, r)) + 1;
                diff = colex_unrank(read_tb(br, comb(r, d)), d);
            }
            uint64_t one = q.value | (1ull << shift(rpos[diff[0]]));
            for (int i = 0; i < r; i++)
                if (i != diff[0]) one |= uint64_t(br.get_bool()) << shift(rpos[i]);
            uint64_t mask = 0;
            for (int i : diff) mask |= 1ull << shift(rpos[i]);
            out.leaves.push_back({one, (q.code << 1) | 1, q.len + 1});
            out.leaves.push_back({one ^ mask, q.code << 1, q.len + 1});
            continue;
        }
        uint64_t idx = read_tb(br, r);
        int p = -1;
        if (pos_mode == POS_MSB) {
            p = rpos[idx];
        } else {
            for (int c : pos_order)
                if ((q.rem >> c) & 1 && idx-- == 0) {
                    p = c;
                    break;
                }
        }
        uint64_t crem = q.rem & ~(1ull << p);
        if (k == STRAIT) {
            queue.push_back({crem, q.value | (uint64_t(br.get_bool()) << shift(p)), q.code, q.len});
        } else {
            queue.push_back({crem, q.value | (1ull << shift(p)), (q.code << 1) | 1, q.len + 1});
            queue.push_back({crem, q.value, q.code << 1, q.len + 1});
        }
    }
    if (out.leaves.empty()) throw std::runtime_error("empty tree");
    return out;
}

std::vector<uint8_t> encode_pair(const uint64_t* x, size_t n, int width, int mask_mode, int stream, int pos,
                                 PairStats* stats) {
    if (n == 0) throw std::runtime_error("empty input");
    if (n >= (1ull << 32)) throw std::runtime_error("item count does not fit the 32-bit header field");
    check_width(x, n, width);
    Uniq u = unique_counts(x, n, width);
    Tree tree = build_tree(u.values, width);
    ValueIndex vi(width);
    vi.assign(u.values);

    struct Choice {
        TreeOut tree;
        int mask, stream;
        std::vector<int> lengths;
        uint64_t stream_bits, length_bits, total;
    };
    Choice best{};
    bool have = false;
    for (int mode : {RAW, COMB}) {
        if (mask_mode != MASK_AUTO && mask_mode != mode) continue;
        Choice c{write_pair_tree(tree, mode, pos), mode, 0, {}, 0, 0, 0};
        std::vector<uint64_t> cnt;
        for (const Leaf& l : c.tree.leaves) cnt.push_back(u.counts[vi[l.value]]);
        c.lengths = huffman_lengths(cnt);
        uint64_t huff = 0, tree_bits = 0;
        for (size_t i = 0; i < cnt.size(); i++) {
            huff += cnt[i] * c.lengths[i];
            tree_bits += cnt[i] * c.tree.leaves[i].len;
        }
        uint64_t lbits = code_length_bits(c.lengths);
        c.stream = stream == STREAM_AUTO ? (huff + lbits < tree_bits ? HUFFMAN : TREE_CODES) : stream;
        c.stream_bits = c.stream == HUFFMAN ? huff : tree_bits;
        c.length_bits = c.stream == HUFFMAN ? lbits : 0;
        c.total = 33 + c.tree.bw.bits() + c.stream_bits + c.length_bits;
        if (!have || c.total < best.total) {
            best = std::move(c);
            have = true;
        }
    }

    BitWriter bw;
    bw.put_int(n, 32);
    bw.append(best.tree.bw);
    bw.put_int(best.stream, 1);
    std::vector<uint64_t> codes;
    std::vector<int> lens;
    if (best.stream == HUFFMAN) {
        put_code_lengths(bw, best.lengths);
        codes = canonical_codes(best.lengths);
        lens = best.lengths;
    } else {
        for (const Leaf& l : best.tree.leaves) {
            codes.push_back(l.code);
            lens.push_back(l.len);
        }
    }
    std::vector<uint64_t> leaf_values;
    for (const Leaf& l : best.tree.leaves) leaf_values.push_back(l.value);
    vi.assign(leaf_values);
    for (size_t k = 0; k < n; k++) {
        uint32_t i = vi[x[k]];
        bw.put_int(codes[i], lens[i]);
    }
    if (bw.bits() != best.total) throw std::logic_error("size mismatch");
    if (stats) {
        stats->header = 33;
        stats->tree = best.tree.bw.bits();
        stats->code_lengths = best.length_bits;
        stats->stream = best.stream_bits;
        stats->mask_mode = best.mask;
        stats->stream_mode = best.stream;
    }
    return bw.bytes();
}

namespace {

void decode_pair(const uint8_t* data, size_t size, int* width, const std::function<void(uint64_t)>& begin,
                 const SymSink& sink) {
    BitReader br(data, size);
    uint64_t n = br.get_int(32);
    ReadTree rt = read_pair_tree(br, n);
    if (width) *width = rt.width;
    begin(n);
    std::vector<uint64_t> codes;
    std::vector<int> lens;
    if (br.get_int(1) == HUFFMAN) {
        lens = read_code_lengths(br, rt.leaves.size());
        codes = canonical_codes(lens);
    } else {
        for (const Leaf& l : rt.leaves) {
            codes.push_back(l.code);
            lens.push_back(l.len);
        }
    }
    std::vector<uint64_t> leaves;
    for (const Leaf& l : rt.leaves) leaves.push_back(l.value);
    decode_chunks(br, PrefixDecoder(codes, lens), 0, n, leaves, sink);
}

}  // namespace

std::vector<uint64_t> decode_pair(const uint8_t* data, size_t size, int* width) {
    std::vector<uint64_t> out;
    decode_pair(data, size, width, [&](uint64_t n) { out.resize(n); },
                [&](uint64_t start, const uint32_t* sym, size_t m, const std::vector<uint64_t>& leaves) {
                    for (size_t i = 0; i < m; i++) out[start + i] = leaves[sym[i]];
                });
    return out;
}

std::string decode_pair_le(const uint8_t* data, size_t size, int bytes) {
    std::string out;
    decode_pair(data, size, nullptr, [&](uint64_t n) { out.resize(n * bytes); },
                [&](uint64_t start, const uint32_t* sym, size_t m, const std::vector<uint64_t>& leaves) {
                    put_le(leaves, sym, m, bytes, &out[start * bytes]);
                });
    return out;
}

}  // namespace cle
