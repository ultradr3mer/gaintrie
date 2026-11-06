#pragma once
#include <cstdint>
#include <vector>

#include "bitio.hpp"

namespace cle {

constexpr int MAX_LEN_BITS = 6;

inline int pos_bits(uint64_t n) { return n > 1 ? 64 - __builtin_clzll(n - 1) : 0; }
inline int popcount(uint64_t v) { return __builtin_popcountll(v); }

int tb_bits(uint64_t idx, uint64_t n);
void put_tb(BitWriter& bw, uint64_t idx, uint64_t n);
uint64_t read_tb(BitReader& br, uint64_t n);

uint64_t comb(int n, int k);
uint64_t colex_rank(const std::vector<int>& idx);
std::vector<int> colex_unrank(uint64_t rank, int count);

std::vector<int> huffman_lengths(const std::vector<uint64_t>& counts);
std::vector<uint64_t> canonical_codes(const std::vector<int>& lengths);

uint64_t code_length_bits(const std::vector<int>& lengths);
void put_code_lengths(BitWriter& bw, const std::vector<int>& lengths);
std::vector<int> read_code_lengths(BitReader& br, size_t count);

// prefix code -> symbol: first-level table over `K` bits, trie below
class PrefixDecoder {
public:
    PrefixDecoder(const std::vector<uint64_t>& codes, const std::vector<int>& lens);
    void decode(BitReader& br, size_t n, uint32_t* out) const;

private:
    struct Entry {
        int32_t v;  // >= 0 symbol, < 0: -(trie node) to continue from, INT32_MIN: invalid
        uint8_t len;
    };
    int k_ = 0;
    bool single_ = false;
    std::vector<int32_t> child_;  // 2 per node; 0 none, > 0 node, < 0 -(symbol + 1)
    std::vector<Entry> table_;
};

}  // namespace cle
