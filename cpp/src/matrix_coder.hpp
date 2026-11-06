#pragma once
#include <cstdint>
#include <vector>

namespace cle {

enum MatrixMode { GLOBAL = 0, HYBRID = 1 };

struct MatrixStats {
    uint64_t header = 0, framing = 0, global_tree = 0, global_lengths = 0, bin_trees = 0, stream = 0;
    uint64_t total() const { return header + framing + global_tree + global_lengths + bin_trees + stream; }
};

std::vector<uint8_t> encode_matrix(const std::vector<uint64_t>& x, int width, int mode, uint64_t bin_size = 1 << 20,
                                   MatrixStats* stats = nullptr);
std::vector<uint64_t> decode_matrix(const uint8_t* data, size_t size, int* width = nullptr);

}  // namespace cle
