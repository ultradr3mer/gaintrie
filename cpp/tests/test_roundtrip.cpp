#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "code_utils.hpp"
#include "matrix_coder.hpp"
#include "pair_writer.hpp"

using namespace cle;
using Vec = std::vector<uint64_t>;

static int failures = 0;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            failures++;                                               \
            std::fprintf(stderr, "FAIL %s:%d %s ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                        \
            std::fprintf(stderr, "\n");                               \
        }                                                             \
    } while (0)

static std::vector<uint8_t> unhex(const std::string& h) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i < h.size(); i += 2) out.push_back(uint8_t(std::stoi(h.substr(i, 2), nullptr, 16)));
    return out;
}

struct Case {
    Vec x;
    int width;
};

static std::vector<Case> cases() {
    std::mt19937_64 rng(3);
    std::vector<Case> out = {{{0, 2}, 2}, {{2, 3, 4, 5, 5, 2}, 3}, {{0, 1}, 3}, {{1, 4, 6, 7, 9, 12, 200, 201, 255, 4, 4}, 8},
                             {{7}, 5}, {{0, 1, 1, 0, 1}, 1}, {{5, 5, 5}, 3}};
    Vec a;
    for (int i = 0; i < 300; i++) a.push_back(rng() & 0xffff);
    out.push_back({a, 16});
    a.clear();
    for (int i = 0; i < 500; i++) a.push_back(rng() & 0xffffffffull);
    out.push_back({a, 32});
    a.clear();
    std::geometric_distribution<int> geo(0.05);
    for (int i = 0; i < 2000; i++) a.push_back(uint64_t(geo(rng) + 1) % 4096);
    out.push_back({a, 12});
    a.clear();
    for (int i = 0; i < 400; i++) a.push_back(rng());
    out.push_back({a, 64});
    out.push_back({{0, 1, (1ull << 62) | 5, (1ull << 63) - 1, (1ull << 62) | 5, ~0ull, 1ull << 63}, 64});
    a.clear();
    std::geometric_distribution<int> geo2(0.02);
    for (int i = 0; i < 20000; i++) a.push_back(uint64_t(geo2(rng)) % 3000);
    out.push_back({a, 16});
    a.assign(300, 5);
    for (int i = 0; i < 300; i++) a.push_back(rng() % 50);
    out.push_back({a, 8});
    return out;
}

static void test_known_answers() {
    // written by the Python reference (pair_writer.best_writer / matrix_coder.encode_matrix)
    struct Kat {
        Case c;
        const char* pair;
        const char* global;
        const char* hybrid3;
    };
    std::vector<Kat> kats = {
        {{{0, 2}, 2}, "00000002043840", "0000000204002000000000002a08701000000028",
         "0000000206000000060000002a087010000000f0438000000050"},
        {{{1, 4, 6, 7, 9, 12, 200, 201, 255, 4, 4}, 8}, "0000000b1ed2e22090b06fe626b885408ce565d100",
         "0000000b1c00200000000000e63da5c4412160dfcc4d710a84ffaa40000008911fdedc00",
         "0000000b1e00000006000000e63da5c4412160dfcc4d710a84ffaa4000000c8726000001f070000000598000001a8e6960000acfb8"
         "00000016c0000006c39a584536dfc400000002b80000005070810000000000"},
        {{{2, 3, 4, 5, 5, 2}, 3}, "00000006087311ac58", "0000000608002000000000004610e6230bc000000313a4",
         "000000060a000000060000004610e6230bc0000007022394b800000017800000020147a000000064"},
    };
    for (const Kat& k : kats) {
        auto p = unhex(k.pair), g = unhex(k.global), h = unhex(k.hybrid3);
        CHECK(decode_pair(p.data(), p.size()) == k.c.x, "py pair decode w=%d", k.c.width);
        CHECK(decode_matrix(g.data(), g.size()) == k.c.x, "py global decode w=%d", k.c.width);
        CHECK(decode_matrix(h.data(), h.size()) == k.c.x, "py hybrid decode w=%d", k.c.width);
        CHECK(encode_pair(k.c.x, k.c.width) == p, "pair bytes w=%d", k.c.width);
        CHECK(encode_matrix(k.c.x, k.c.width, GLOBAL) == g, "global bytes w=%d", k.c.width);
        CHECK(encode_matrix(k.c.x, k.c.width, HYBRID, 3) == h, "hybrid bytes w=%d", k.c.width);
    }
}

static void test_pair_round_trip() {
    for (const Case& c : cases()) {
        for (int mask : {RAW, COMB, MASK_AUTO}) {
            uint64_t best[3] = {};
            for (int stream : {TREE_CODES, HUFFMAN, STREAM_AUTO}) {
                uint64_t sizes[3] = {};
                for (int pos : {POS_MSB, POS_ORDER, POS_AUTO}) {
                    PairStats st;
                    auto d = encode_pair(c.x, c.width, mask, stream, pos, &st);
                    CHECK((st.total() + 7) / 8 == d.size(), "size w=%d", c.width);
                    int w = 0;
                    CHECK(decode_pair(d.data(), d.size(), &w) == c.x, "w=%d mask=%d stream=%d pos=%d", c.width, mask,
                          stream, pos);
                    CHECK(w == c.width, "width %d", w);
                    sizes[pos] = st.total();
                }
                CHECK(sizes[POS_AUTO] == std::min(sizes[POS_MSB], sizes[POS_ORDER]), "pos auto w=%d", c.width);
                best[stream] = sizes[POS_AUTO];
            }
            CHECK(best[STREAM_AUTO] == std::min(best[TREE_CODES], best[HUFFMAN]), "stream auto w=%d", c.width);
        }
    }
}

static void test_matrix_round_trip() {
    for (const Case& c : cases()) {
        for (auto [mode, size] : std::vector<std::pair<int, uint64_t>>{
                 {GLOBAL, 1 << 20}, {HYBRID, 1}, {HYBRID, 3}, {HYBRID, 64}, {HYBRID, 300}, {HYBRID, 1 << 20}}) {
            MatrixStats st;
            auto d = encode_matrix(c.x, c.width, mode, size, &st);
            CHECK((st.total() + 7) / 8 == d.size(), "size");
            int w = 0;
            CHECK(decode_matrix(d.data(), d.size(), &w) == c.x, "w=%d mode=%d bins=%llu", c.width, mode,
                  (unsigned long long)size);
            CHECK(w == c.width, "width %d", w);
        }
    }
}

static void test_code_utils() {
    for (uint64_t n : std::vector<uint64_t>{1, 2, 3, 5, 7, 8, 13, 64, 1000, comb(64, 32)}) {
        BitWriter bw;
        std::vector<uint64_t> idx;
        for (uint64_t i = 0; i < std::min<uint64_t>(n, 300); i++) idx.push_back(i);
        idx.push_back(n - 1);
        uint64_t bits = 0;
        for (uint64_t i : idx) {
            put_tb(bw, i, n);
            bits += tb_bits(i, n);
        }
        CHECK(bw.bits() == bits, "tb bits n=%llu", (unsigned long long)n);
        auto bytes = bw.bytes();
        BitReader br(bytes.data(), bytes.size());
        for (uint64_t i : idx) CHECK(read_tb(br, n) == i, "tb n=%llu", (unsigned long long)n);
    }
    for (int r : {1, 5, 16, 64})
        for (int d = 1; d <= std::min(r, 4); d++)
            for (uint64_t rank : std::vector<uint64_t>{0, 1, comb(r, d) / 2, comb(r, d) - 1}) {
                if (rank >= comb(r, d)) continue;
                auto diff = colex_unrank(rank, d);
                CHECK(colex_rank(diff) == rank && diff.back() < r, "colex r=%d d=%d", r, d);
            }
    std::mt19937_64 rng(5);
    for (int n : {1, 2, 3, 17, 300}) {
        std::vector<uint64_t> counts;
        std::geometric_distribution<int> geo(0.05);
        for (int i = 0; i < n; i++) counts.push_back(geo(rng) + 1);
        auto lengths = huffman_lengths(counts);
        auto codes = canonical_codes(lengths);
        PrefixDecoder dec(codes, lengths);  // throws if not prefix-free
        CHECK(int(lengths.size()) == n, "huffman n=%d", n);
    }
    BitWriter bw;
    bw.put_int(0x123456789abcdefull, 61);
    bw.put(true);
    bw.put_int(5, 3);
    auto b = bw.bytes();
    BitReader br(b.data(), b.size());
    CHECK(br.get_int(61) == 0x123456789abcdefull && br.get_bool() && br.get_int(3) == 5, "bitio");
}

int main() {
    test_code_utils();
    test_known_answers();
    test_pair_round_trip();
    test_matrix_round_trip();
    if (failures) {
        std::fprintf(stderr, "%d checks failed\n", failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
