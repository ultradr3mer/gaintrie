#include "sha256.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>
#include <immintrin.h>
#define CLE_SHA_NI 1
#endif

namespace cle {
namespace {

const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void blocks_portable(uint32_t* h, const uint8_t* p, size_t n) {
    uint32_t w[64];
    for (; n--; p += 64) {
        for (int i = 0; i < 16; i++)
            w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    }
}

#ifdef CLE_SHA_NI
__attribute__((target("sha,sse4.1,ssse3"))) void blocks_ni(uint32_t* h, const uint8_t* p, size_t n) {
    const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i tmp = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h)), 0xB1);
    __m128i s1 = _mm_shuffle_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h + 4)), 0x1B);
    __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);
    s1 = _mm_blend_epi16(s1, tmp, 0xF0);
    for (; n--; p += 64) {
        __m128i abef = s0, cdgh = s1, w[4];
#pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            __m128i& m = w[i & 3];
            if (i < 4) {
                m = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16 * i)), MASK);
            } else {
                __m128i t = _mm_sha256msg1_epu32(w[i & 3], w[(i + 1) & 3]);
                t = _mm_add_epi32(t, _mm_alignr_epi8(w[(i + 3) & 3], w[(i + 2) & 3], 4));
                m = _mm_sha256msg2_epu32(t, w[(i + 3) & 3]);
            }
            __m128i msg = _mm_add_epi32(m, _mm_loadu_si128(reinterpret_cast<const __m128i*>(K + 4 * i)));
            s1 = _mm_sha256rnds2_epu32(s1, s0, msg);
            s0 = _mm_sha256rnds2_epu32(s0, s1, _mm_shuffle_epi32(msg, 0x0E));
        }
        s0 = _mm_add_epi32(s0, abef);
        s1 = _mm_add_epi32(s1, cdgh);
    }
    tmp = _mm_shuffle_epi32(s0, 0x1B);
    s1 = _mm_shuffle_epi32(s1, 0xB1);
    s0 = _mm_blend_epi16(tmp, s1, 0xF0);
    s1 = _mm_alignr_epi8(s1, tmp, 8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(h), s0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(h + 4), s1);
}

bool has_sha_ni() {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & bit_SSE4_1) || !(c & bit_SSSE3)) return false;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return b & (1u << 29);
}

const bool SHA_NI = has_sha_ni() && !std::getenv("GAINTRIE_NO_SHA_NI");
#endif

}  // namespace

Sha256::Sha256() {
    const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, init, sizeof h_);
}

void Sha256::blocks(const uint8_t* p, size_t n) {
#ifdef CLE_SHA_NI
    if (SHA_NI) return blocks_ni(h_, p, n);
#endif
    blocks_portable(h_, p, n);
}

void Sha256::update(const void* data, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    total_ += size;
    if (nbuf_) {
        size_t k = std::min(size, 64 - nbuf_);
        std::memcpy(buf_ + nbuf_, p, k);
        nbuf_ += k, p += k, size -= k;
        if (nbuf_ < 64) return;
        blocks(buf_, 1);
        nbuf_ = 0;
    }
    blocks(p, size / 64);
    p += size / 64 * 64;
    nbuf_ = size % 64;
    std::memcpy(buf_, p, nbuf_);
}

std::string Sha256::hexdigest() {
    uint64_t bits = total_ * 8;
    uint8_t pad[128] = {0x80};
    size_t k = (nbuf_ < 56 ? 56 : 120) - nbuf_;
    for (int i = 0; i < 8; i++) pad[k + i] = uint8_t(bits >> (56 - 8 * i));
    update(pad, k + 8);
    static const char* HEX = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h_)
        for (int s = 28; s >= 0; s -= 4) out += HEX[(v >> s) & 15];
    return out;
}

std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    Sha256 h;
    std::vector<char> buf(size_t(1) << 22);
    while (f) {
        f.read(buf.data(), std::streamsize(buf.size()));
        h.update(buf.data(), size_t(f.gcount()));
    }
    return h.hexdigest();
}

}  // namespace cle
