#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "matrix_coder.hpp"
#include "pair_writer.hpp"

using namespace cle;

namespace {

const char* USAGE =
    "usage:\n"
    "  gaintrie encode --format pair|matrix --width W [--mode global|hybrid] [--bin-size N]\n"
    "                  [--mask auto|raw|comb] [--stream auto|tree|huffman] [--pos auto|msb|order]\n"
    "                  [--in-bytes 1|2|4|8] [--stats] in.bin out.cle\n"
    "  gaintrie decode --format pair|matrix [--out-bytes 1|2|4|8] [--stats] in.cle out.bin\n"
    "values are little-endian unsigned; default element size = smallest of 1/2/4/8 bytes holding W bits\n";

int default_bytes(int width) { return width <= 8 ? 1 : width <= 16 ? 2 : width <= 32 ? 4 : 8; }

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void write_file(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(reinterpret_cast<const char*>(data), std::streamsize(size));
    if (!f) throw std::runtime_error("write failed: " + path);
}

std::vector<uint64_t> unpack(const std::vector<uint8_t>& raw, int bytes) {
    if (raw.size() % bytes) throw std::runtime_error("input size is not a multiple of the element size");
    std::vector<uint64_t> x(raw.size() / bytes);
    for (size_t i = 0; i < x.size(); i++) {
        uint64_t v = 0;
        for (int b = bytes - 1; b >= 0; b--) v = (v << 8) | raw[i * bytes + b];
        x[i] = v;
    }
    return x;
}

std::vector<uint8_t> pack(const std::vector<uint64_t>& x, int bytes) {
    std::vector<uint8_t> raw(x.size() * bytes);
    for (size_t i = 0; i < x.size(); i++) {
        if (bytes < 8 && x[i] >> (8 * bytes)) throw std::runtime_error("value does not fit the output element size");
        for (int b = 0; b < bytes; b++) raw[i * bytes + b] = uint8_t(x[i] >> (8 * b));
    }
    return raw;
}

int pick(const std::map<std::string, std::string>& opt, const std::string& key,
         const std::map<std::string, int>& names, int def) {
    auto it = opt.find(key);
    if (it == opt.end()) return def;
    auto n = names.find(it->second);
    if (n == names.end()) throw std::runtime_error("invalid value for --" + key + ": " + it->second);
    return n->second;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << USAGE;
            return 2;
        }
        std::string cmd = argv[1];
        std::map<std::string, std::string> opt;
        std::vector<std::string> files;
        bool stats = false;
        for (int i = 2; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--stats") {
                stats = true;
            } else if (a.rfind("--", 0) == 0) {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                opt[a.substr(2)] = argv[++i];
            } else {
                files.push_back(a);
            }
        }
        if ((cmd != "encode" && cmd != "decode") || files.size() != 2 || !opt.count("format")) {
            std::cerr << USAGE;
            return 2;
        }
        int format = pick(opt, "format", {{"pair", 0}, {"matrix", 1}}, 0);
        auto t0 = std::chrono::steady_clock::now();
        if (cmd == "encode") {
            if (!opt.count("width")) throw std::runtime_error("--width is required");
            int width = std::stoi(opt["width"]);
            if (width < 1 || width > 64) throw std::runtime_error("width must be 1..64");
            int bytes = opt.count("in-bytes") ? std::stoi(opt["in-bytes"]) : default_bytes(width);
            if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8)
                throw std::runtime_error("--in-bytes must be 1, 2, 4 or 8");
            std::vector<uint64_t> x = unpack(read_file(files[0]), bytes);
            auto t1 = std::chrono::steady_clock::now();
            std::vector<uint8_t> data;
            char line[512];
            if (format == 0) {
                PairStats st;
                data = encode_pair(x, width, pick(opt, "mask", {{"auto", MASK_AUTO}, {"raw", RAW}, {"comb", COMB}}, MASK_AUTO),
                                   pick(opt, "stream", {{"auto", STREAM_AUTO}, {"tree", TREE_CODES}, {"huffman", HUFFMAN}},
                                        STREAM_AUTO),
                                   pick(opt, "pos", {{"auto", POS_AUTO}, {"msb", POS_MSB}, {"order", POS_ORDER}}, POS_AUTO),
                                   &st);
                std::snprintf(line, sizeof line,
                              "header=%llu tree=%llu code_lengths=%llu stream=%llu mask=%s stream_mode=%s",
                              (unsigned long long)st.header, (unsigned long long)st.tree,
                              (unsigned long long)st.code_lengths, (unsigned long long)st.stream,
                              st.mask_mode == RAW ? "raw" : "comb", st.stream_mode == HUFFMAN ? "huffman" : "tree");
            } else {
                MatrixStats st;
                int mode = pick(opt, "mode", {{"global", GLOBAL}, {"hybrid", HYBRID}}, GLOBAL);
                uint64_t bin_size = opt.count("bin-size") ? std::stoull(opt["bin-size"]) : (1ull << 20);
                data = encode_matrix(x, width, mode, bin_size, &st);
                std::snprintf(line, sizeof line,
                              "header=%llu framing=%llu global_tree=%llu global_lengths=%llu bin_trees=%llu stream=%llu",
                              (unsigned long long)st.header, (unsigned long long)st.framing,
                              (unsigned long long)st.global_tree, (unsigned long long)st.global_lengths,
                              (unsigned long long)st.bin_trees, (unsigned long long)st.stream);
            }
            double t_enc = seconds_since(t1);
            write_file(files[1], data.data(), data.size());
            if (stats)
                std::fprintf(stderr, "encode n=%zu width=%d bytes=%zu bits_per_value=%.4f time=%.4f total_time=%.4f %s\n",
                             x.size(), width, data.size(), 8.0 * data.size() / x.size(), t_enc, seconds_since(t0), line);
        } else {
            std::vector<uint8_t> data = read_file(files[0]);
            auto t1 = std::chrono::steady_clock::now();
            int width = 0;
            std::vector<uint64_t> x = format == 0 ? decode_pair(data.data(), data.size(), &width)
                                                  : decode_matrix(data.data(), data.size(), &width);
            double t_dec = seconds_since(t1);
            int bytes = opt.count("out-bytes") ? std::stoi(opt["out-bytes"]) : default_bytes(width);
            if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8)
                throw std::runtime_error("--out-bytes must be 1, 2, 4 or 8");
            std::vector<uint8_t> raw = pack(x, bytes);
            write_file(files[1], raw.data(), raw.size());
            if (stats)
                std::fprintf(stderr, "decode n=%zu width=%d time=%.4f total_time=%.4f\n", x.size(), width, t_dec,
                             seconds_since(t0));
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gaintrie: " << e.what() << "\n";
        return 1;
    }
}
