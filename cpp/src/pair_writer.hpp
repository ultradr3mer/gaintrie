#pragma once
#include <cstdint>
#include <vector>

#include "bitio.hpp"
#include "gain_coder.hpp"

namespace cle {

enum MaskMode { RAW = 0, COMB = 1, MASK_AUTO = 2 };
enum StreamMode { TREE_CODES = 0, HUFFMAN = 1, STREAM_AUTO = 2 };
enum PosMode { POS_MSB = 0, POS_ORDER = 1, POS_AUTO = 2 };

struct Leaf {
    uint64_t value;
    uint64_t code;  // tree path
    int len;
};

// a written tree; leaves in the order the decoder rebuilds them
struct TreeOut {
    BitWriter bw;
    std::vector<Leaf> leaves;
    int width = 0;
};

class PairWriter {
public:
    PairWriter(const Tree& tree, int mask_mode);

    struct Entry {
        int node;
        uint64_t rem;
        uint64_t code;
        int len;
        int kind;
    };
    std::vector<Entry> walk() const;
    TreeOut write_tree(int pos_mode) const;

private:
    struct Codes {
        uint64_t bits[4] = {};
        int len[4] = {};
        bool operator==(const Codes& o) const;
    };
    static Codes kind_codes(const std::vector<int>& order);
    static std::vector<int> most_common(const std::vector<Entry>& nodes);

    void two_leaves(int node, uint64_t& a, uint64_t& b) const;
    std::vector<int> pair_diff(uint64_t a, uint64_t b, uint64_t rem) const;
    uint64_t mask_bits(int r, const std::vector<int>& diff) const;
    uint64_t pair_cost(int node, uint64_t rem, const Codes& codes) const;
    uint64_t classic_cost(int node, uint64_t rem, const Codes& codes) const;
    void heads(int node, uint64_t rem, std::vector<std::pair<int, uint64_t>>& out) const;
    void choose();
    std::vector<int> position_order(const std::vector<Entry>& nodes) const;

    uint64_t pmask(uint64_t v) const;
    int bit(uint64_t v, int p) const { return int((v >> (width_ - 1 - p)) & 1); }

    const Tree& tree_;
    int width_;
    int mask_mode_;
    std::vector<uint8_t> pair_;
};

// tree only (no item count), as pair_writer.write_tree / read_tree
TreeOut write_pair_tree(const Tree& tree, int mask_mode, int pos_mode = POS_AUTO);
TreeOut best_tree(const Tree& tree, int pos_mode = POS_AUTO);  // smaller of RAW / COMB
struct ReadTree {
    int width;
    std::vector<Leaf> leaves;
};
ReadTree read_pair_tree(BitReader& br, uint64_t max_leaves);

struct PairStats {
    uint64_t header = 0, tree = 0, code_lengths = 0, stream = 0;
    int mask_mode = 0, stream_mode = 0;
    uint64_t total() const { return header + tree + code_lengths + stream; }
};

std::vector<uint8_t> encode_pair(const std::vector<uint64_t>& x, int width, int mask_mode = MASK_AUTO,
                                 int stream = STREAM_AUTO, int pos = POS_AUTO, PairStats* stats = nullptr);
std::vector<uint64_t> decode_pair(const uint8_t* data, size_t size, int* width = nullptr);

}  // namespace cle
