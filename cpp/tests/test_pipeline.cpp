#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "json.hpp"
#include "pipeline.hpp"
#include "sha256.hpp"
#include "util.hpp"

using namespace cle;
namespace fs = std::filesystem;

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

static std::string sha(const std::string& s) {
    Sha256 h;
    h.update(s.data(), s.size());
    return h.hexdigest();
}

static void test_sha256() {
    CHECK(sha("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    CHECK(sha("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    CHECK(sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "448 bits");
    std::string a(1000000, 'a');
    CHECK(sha(a) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "million a");
    std::mt19937_64 rng(5);
    std::string r(100003, '\0');
    for (char& c : r) c = char(rng());
    std::string whole = sha(r);
    for (size_t step : {1, 7, 63, 64, 65, 1000}) {
        Sha256 h;
        for (size_t i = 0; i < r.size(); i += step) h.update(r.data() + i, std::min(step, r.size() - i));
        CHECK(h.hexdigest() == whole, "chunks of %zu", step);
    }
}

static void test_base64() {
    CHECK(base64_encode("Man") == "TWFu", "Man");
    CHECK(base64_encode("Ma") == "TWE=", "Ma");
    CHECK(base64_encode("M") == "TQ==", "M");
    CHECK(base64_encode("") == "", "empty");
    std::string s;
    for (int i = 0; i < 300; i++) s += char(i * 37);
    for (size_t n = 0; n < 10; n++) CHECK(base64_decode(base64_encode(s.substr(n))) == s.substr(n), "n=%zu", n);
}

static void test_json() {
    struct F {
        double x;
        const char* s;
    };
    for (F f : {F{0.0, "0.0"}, F{-0.0, "-0.0"}, F{1.0, "1.0"}, F{0.1, "0.1"}, F{1e-05, "1e-05"}, F{0.0001, "0.0001"},
                F{1e16, "1e+16"}, F{1234567890123456.0, "1234567890123456.0"}, F{1.0 / 3, "0.3333333333333333"},
                F{2.5e-07, "2.5e-07"}, F{10.5217, "10.5217"}, F{0.6619810141077744, "0.6619810141077744"},
                F{1.5e300, "1.5e+300"}})
        CHECK(py_float(f.x) == f.s, "%s -> %s", f.s, py_float(f.x).c_str());
    CHECK(py_round(0.00285, 4) == 0.0028 || py_round(0.00285, 4) == 0.0029, "round");
    CHECK(py_round(2.675, 2) == 2.67, "round half: %s", py_float(py_round(2.675, 2)).c_str());

    std::string text = R"({"b": [1, -2, 3.5, true, false, null], "a": {"x": "\u00e9\ud83d\ude00\"\\\n"}, "e": [], "o": {}})";
    Json v = json_parse(text);
    CHECK(v.obj.size() == 4 && v.obj[0].first == "b" && v.obj[1].first == "a", "key order");
    CHECK(v.at("a").at("x").s == "\xc3\xa9\xf0\x9f\x98\x80\"\\\n", "unicode");
    CHECK(v.at("b").arr[1].i == -2 && v.at("b").arr[2].f == 3.5, "numbers");
    CHECK(json_dump(v) == text, "dump: %s", json_dump(v).c_str());
    Json o = Json::object();
    o.set("k", 1);
    Json& l = o.set("l", Json::array());
    l.push(Json(nullptr));
    Json in = Json::object();
    in.set("t", true);
    l.push(in);
    CHECK(json_dump(o, 1) == "{\n \"k\": 1,\n \"l\": [\n  null,\n  {\n   \"t\": true\n  }\n ]\n}", "indent: %s",
          json_dump(o, 1).c_str());
}

// as tests/test_pipeline.py write_shard
struct T {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::string data;
};

static void write_shard(const fs::path& path, const std::vector<T>& tensors, bool metadata) {
    Json h = Json::object();
    if (metadata) {
        Json m = Json::object();
        m.set("format", "pt");
        h.set("__metadata__", m);
    }
    std::string blobs;
    for (const T& t : tensors) {
        Json e = Json::object(), shape = Json::array(), offs = Json::array();
        for (int64_t d : t.shape) shape.push(Json(static_cast<long long>(d)));
        offs.push(Json(blobs.size()));
        offs.push(Json(blobs.size() + t.data.size()));
        e.set("dtype", t.dtype);
        e.set("shape", shape);
        e.set("data_offsets", offs);
        h.set(t.name, e);
        blobs += t.data;
    }
    std::string hs = json_dump(h);
    hs.append((8 - hs.size() % 8) % 8, ' ');
    std::ofstream f(path, std::ios::binary);
    uint64_t n = hs.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f << hs << blobs;
}

static std::mt19937_64 rng(11);

template <class U>
static std::string bytes_of(const std::vector<U>& v) {
    return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(U));
}

static std::string bf16_like(size_t n, double mean, double sd) {
    std::normal_distribution<float> d{float(mean), float(sd)};
    std::vector<uint16_t> v(n);
    for (auto& x : v) {
        float f = d(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        x = uint16_t(u >> 16);
    }
    return bytes_of(v);
}

static std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static void test_roundtrip(const fs::path& tmp) {
    fs::path src = tmp / "src", run = tmp / "run";
    fs::create_directories(src);
    std::normal_distribution<float> nf(0, 1);
    std::vector<float> scale(5000);
    for (float& f : scale) f = nf(rng);
    std::vector<int64_t> pos(10);
    for (int i = 0; i < 10; i++) pos[i] = i;
    std::vector<uint16_t> constant(2000, 0x3F80);
    std::vector<uint8_t> mask(77);
    for (auto& b : mask) b = uint8_t(rng() & 1);
    write_shard(src / "model-00001-of-00002.safetensors",
                {{"model.layers.0.input_layernorm.weight", "BF16", {4096}, bf16_like(4096, 1, 0.1)},
                 {"model.layers.0.self_attn.k_proj.weight", "BF16", {64, 3000}, bf16_like(192000, 0, 0.02)},
                 {"model.layers.0.self_attn.q_norm.weight", "BF16", {128}, bf16_like(128, 0, 0.02)},
                 {"model.empty", "BF16", {0}, ""}},
                true);
    write_shard(src / "model-00002-of-00002.safetensors",
                {{"model.layers.1.mlp.gate_proj.weight", "BF16", {300, 1000}, bf16_like(300000, 0, 0.02)},
                 {"model.scale", "F32", {5000}, bytes_of(scale)},
                 {"model.position_ids", "I64", {10}, bytes_of(pos)},
                 {"model.constant", "BF16", {2000}, bytes_of(constant)},
                 {"model.mask", "BOOL", {77}, bytes_of(mask)}},
                false);
    std::ofstream(src / "notes.txt") << "not a shard";

    fs::path archive = run / "model.clar";
    CompressResult comp = pipeline_compress(src.string(), archive.string(), run.string(), 1 << 16, 4);
    pipeline_summarize(comp, nullptr, run.string());
    DecompressResult dec = pipeline_decompress(archive.string(), (run / "restored").string(), run.string(), 4);
    for (const char* f : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
        CHECK(read_file(run / "restored" / f) == read_file(src / f), "%s", f);
    CHECK(dec.checks.size() == 2 && dec.lossless(), "checks");
    for (const ShardCheck& c : dec.checks) CHECK(c.matches_archive && c.matches_reference == -1, "%s", c.file.c_str());
    CHECK(comp.archive_bytes < comp.input_bytes, "smaller");
    std::set<std::string> kinds;
    for (const UnitRow& u : comp.units) kinds.insert(u.kind);
    CHECK(kinds.count("raw") && (kinds.count("pair") || kinds.count("matrix")), "kinds");
    for (const char* f : {"compress.csv", "tensors.csv", "decompress.csv", "summary.json"})
        CHECK(fs::exists(run / f), "%s", f);

    // reference hashes: a correct sha256.txt passes, a wrong one fails
    std::ofstream(src / "sha256.txt") << sha256_file((src / "model-00001-of-00002.safetensors").string())
                                      << "  model-00001-of-00002.safetensors\n"
                                      << std::string(64, '0') << "  model-00002-of-00002.safetensors\n";
    DecompressResult ref = pipeline_decompress(archive.string(), (run / "restored2").string(), run.string(), 2,
                                               src.string());
    CHECK(ref.checks[0].matches_reference == 1 && ref.checks[1].matches_reference == 0 && !ref.lossless(), "ref");

    // round trip helper + single worker + bins of 1 value
    CHECK(pipeline_roundtrip(src.string(), (tmp / "run1").string(), 1 << 20, 1) == false, "roundtrip vs bad ref");
    fs::remove(src / "sha256.txt");
    CHECK(pipeline_roundtrip(src.string(), (tmp / "run2").string(), 1 << 20, 3), "roundtrip");
    fs::path tiny = tmp / "tiny";
    fs::create_directories(tiny);
    write_shard(tiny / "a.safetensors", {{"x", "U8", {5}, "\x01\x02\x01\x03\x01"}, {"y", "I32", {2}, bytes_of(std::vector<int32_t>{-1, 7})}},
                false);
    CHECK(pipeline_roundtrip(tiny.string(), (tmp / "run3").string(), 1, 2), "bins of one value");
    CHECK(read_file(tmp / "run3" / "restored" / "a.safetensors") == read_file(tiny / "a.safetensors"), "tiny bytes");

    bool threw = false;
    try {
        std::ofstream(tmp / "bad.clar") << "CLAR2xxxxxxxxxxxx";
        pipeline_decompress((tmp / "bad.clar").string(), (tmp / "bad").string(), (tmp / "bad").string(), 1);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "bad magic");
}

int main() {
    test_sha256();
    test_base64();
    test_json();
    fs::path tmp = fs::temp_directory_path() / ("gaintrie_test_" + std::to_string(std::random_device{}()));
    fs::create_directories(tmp);
    try {
        test_roundtrip(tmp);
    } catch (const std::exception& e) {
        failures++;
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
    }
    fs::remove_all(tmp);
    if (failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("all pipeline tests passed\n");
    return 0;
}
