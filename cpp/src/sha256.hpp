#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace cle {

class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t size);
    std::string hexdigest();  // finishes the hash

private:
    void blocks(const uint8_t* p, size_t n);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t nbuf_ = 0;
    uint64_t total_ = 0;
};

std::string sha256_file(const std::string& path);

}  // namespace cle
