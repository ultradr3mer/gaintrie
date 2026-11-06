#pragma once
#include <cstdint>
#include <vector>

namespace cle {

enum Kind : uint8_t { SPLIT = 0, STRAIT = 1, LEAF = 2, PAIR = 3 };

// positions are MSB-first (0 = MSB); a rem set is a mask with bit p = position p
struct TNode {
    Kind kind = LEAF;
    uint8_t pos = 0;  // split / strait position
    uint8_t bit = 0;  // strait value
    int32_t a = -1;   // split: 1-child, strait: child
    int32_t b = -1;   // split: 0-child
    uint32_t leaves = 1;
    uint64_t value = 0;  // leaf
};

struct Tree {
    int width = 0;
    int root = 0;
    std::vector<TNode> nodes;
};

inline uint64_t full_rem(int width) { return width >= 64 ? ~0ull : (1ull << width) - 1; }

// GainCoder over distinct values: straits, then the max-entropy-gain split
Tree build_tree(std::vector<uint64_t> values, int width);

}  // namespace cle
