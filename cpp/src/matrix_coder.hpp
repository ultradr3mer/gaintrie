#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cle {

enum MatrixMode { GLOBAL = 0, HYBRID = 1 };

struct MatrixStats {
    uint64_t header = 0, framing = 0, global_tree = 0, global_lengths = 0, bin_trees = 0, stream = 0;
    uint64_t total() const { return header + framing + global_tree + global_lengths + bin_trees + stream; }
};

std::vector<uint8_t> encode_matrix(const uint64_t* x, size_t n, int width, int mode, uint64_t bin_size = 1 << 20,
                                   MatrixStats* stats = nullptr);
inline std::vector<uint8_t> encode_matrix(const std::vector<uint64_t>& x, int width, int mode,
                                          uint64_t bin_size = 1 << 20, MatrixStats* stats = nullptr) {
    return encode_matrix(x.data(), x.size(), width, mode, bin_size, stats);
}
std::vector<uint64_t> decode_matrix(const uint8_t* data, size_t size, int* width = nullptr);
// decoded values as little-endian elements of `bytes` bytes
std::string decode_matrix_le(const uint8_t* data, size_t size, int bytes);

}  // namespace cle
