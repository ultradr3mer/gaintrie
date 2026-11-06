#include "pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "json.hpp"
#include "matrix_coder.hpp"
#include "pair_writer.hpp"
#include "sha256.hpp"
#include "util.hpp"

namespace cle {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

const char MAGIC[] = "CLAR1";
const size_t MAGIC_LEN = 5;

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

template <class... A>
std::string fmt(const char* f, A... a) {
    char buf[2048];
    std::snprintf(buf, sizeof buf, f, a...);
    return buf;
}

int codec_width(const std::string& dtype) {
    static const std::map<std::string, int> W = {{"BF16", 16}, {"F16", 16}, {"F32", 32}, {"U8", 8},  {"I8", 8},
                                                 {"U16", 16},  {"I16", 16}, {"U32", 32}, {"I32", 32}};
    auto it = W.find(dtype);
    return it == W.end() ? 0 : it->second;
}

// "model.layers.<N>.<rest>" -> (rest, N), else (name, -1)
std::pair<std::string, int64_t> tensor_type(const std::string& name) {
    const std::string pre = "model.layers.";
    if (name.compare(0, pre.size(), pre) == 0) {
        size_t p = pre.size(), q = p;
        while (q < name.size() && name[q] >= '0' && name[q] <= '9') q++;
        if (q > p && q < name.size() && name[q] == '.' && name.find('\n', q) == std::string::npos)
            return {name.substr(q + 1), std::stoll(name.substr(p, q - p))};
    }
    return {name, -1};
}

uint64_t le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int b = 7; b >= 0; b--) v = (v << 8) | p[b];
    return v;
}

std::string le64_bytes(uint64_t v) {
    std::string s(8, '\0');
    for (int b = 0; b < 8; b++) s[b] = char(v >> (8 * b));
    return s;
}

std::string read_at(const fs::path& path, uint64_t offset, uint64_t len) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    f.seekg(std::streamoff(offset));
    std::string s(len, '\0');
    f.read(s.data(), std::streamsize(len));
    if (uint64_t(f.gcount()) != len) throw std::runtime_error("short read: " + path.string());
    return s;
}

std::vector<uint64_t> read_values(const fs::path& path, uint64_t offset, uint64_t n, int width) {
    int bytes = width / 8;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    f.seekg(std::streamoff(offset));
    std::vector<uint64_t> x(n);
    std::vector<uint8_t> buf;
    const uint64_t chunk = 1 << 20;
    for (uint64_t s = 0; s < n; s += chunk) {
        uint64_t m = std::min(chunk, n - s);
        buf.resize(m * bytes);
        f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size()));
        if (uint64_t(f.gcount()) != buf.size()) throw std::runtime_error("short read: " + path.string());
        const uint8_t* p = buf.data();
        uint64_t* o = x.data() + s;
        if (bytes == 1) {
            for (uint64_t i = 0; i < m; i++) o[i] = p[i];
        } else if (bytes == 2) {
            for (uint64_t i = 0; i < m; i++) o[i] = uint64_t(p[2 * i]) | uint64_t(p[2 * i + 1]) << 8;
        } else {
            for (uint64_t i = 0; i < m; i++)
                o[i] = uint64_t(p[4 * i]) | uint64_t(p[4 * i + 1]) << 8 | uint64_t(p[4 * i + 2]) << 16 |
                       uint64_t(p[4 * i + 3]) << 24;
        }
    }
    return x;
}

std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\r\n") == std::string::npos) return s;
    std::string q = "\"";
    for (char c : s) q += c == '"' ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

void write_csv(const fs::path& path, const std::vector<std::string>& head,
               const std::vector<std::vector<std::string>>& rows) {
    if (rows.empty()) return;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path.string());
    auto line = [&](const std::vector<std::string>& r) {
        for (size_t i = 0; i < r.size(); i++) f << (i ? "," : "") << csv_field(r[i]);
        f << "\r\n";
    };
    line(head);
    for (const auto& r : rows) line(r);
}

std::string layer_str(int64_t layer) { return layer < 0 ? "" : std::to_string(layer); }

std::vector<std::string> unit_csv(const UnitRow& u) {
    return {u.tensor,
            u.type,
            layer_str(u.layer),
            u.kind,
            std::to_string(u.start),
            std::to_string(u.values),
            std::to_string(u.raw_bytes),
            std::to_string(u.comp_bytes),
            u.values ? py_float(py_round(u.comp_bytes * 8.0 / double(u.values), 4)) : "0",
            u.raw_bytes ? py_float(double(u.comp_bytes) / double(u.raw_bytes)) : "0",
            py_float(py_round(u.enc_s, 4))};
}

std::vector<std::string> tensor_csv(const TensorRow& r) {
    return {r.tensor,
            r.type,
            layer_str(r.layer),
            r.dtype,
            std::to_string(r.values),
            std::to_string(r.raw_bytes),
            r.choice,
            std::to_string(r.bins_bytes),
            r.has_bins ? py_float(py_round(r.bins_s, 3)) : "0",
            std::to_string(r.matrix_bytes),
            r.has_matrix ? py_float(py_round(r.matrix_s, 3)) : "0",
            std::to_string(r.stored_bytes),
            r.note};
}

struct Shard {
    fs::path path;
    uint64_t size = 0, base = 0;
    std::string header, sha256;
    std::vector<Json> records;
};

struct Unit {
    std::string kind;
    uint64_t start, values;
    int width;
    std::string blob;
    double enc_s;
};

struct Tensor {
    size_t shard = 0;
    std::string name;
    Json info;
    uint64_t start = 0, end = 0, n = 0;
    int width = 0;  // 0: stored raw
    std::vector<Unit> bins;
    Unit matrix, raw;
    bool matrix_ok = false;
    std::string matrix_error;
    std::atomic<size_t> pending{0};
};

class Compressor {
public:
    Compressor(const fs::path& archive, uint64_t bin_size, Clock::time_point t0)
        : out_(archive, std::ios::binary), bin_size_(bin_size), t0_(t0) {
        if (!out_) throw std::runtime_error("cannot write " + archive.string());
        out_.write(MAGIC, MAGIC_LEN);
    }

    void run(std::vector<Shard>& shards, std::vector<std::unique_ptr<Tensor>>& tensors, int workers) {
        shards_ = &shards;
        total_ = tensors.size();
        std::vector<Tensor*> order;
        for (auto& t : tensors) order.push_back(t.get());
        std::stable_sort(order.begin(), order.end(),
                         [](const Tensor* a, const Tensor* b) { return a->end - a->start > b->end - b->start; });
        ThreadPool pool(workers);
        for (Tensor* t : order) {
            const fs::path& path = shards[t->shard].path;
            uint64_t at = shards[t->shard].base + t->start;
            if (!t->width) {
                t->pending = 1;
                pool.submit([this, t, &path, at] {
                    t->raw = {"raw", 0, t->end - t->start, 8, read_at(path, at, t->end - t->start), 0.0};
                    done(*t);
                });
                continue;
            }
            size_t nbins = (t->n + bin_size_ - 1) / bin_size_;
            t->bins.resize(nbins);
            t->pending = nbins + 1;
            pool.submit([this, t, &path, at] {
                auto t1 = Clock::now();
                std::vector<uint64_t> x = read_values(path, at, t->n, t->width);
                try {  // one stream segment must stay below 2^32 bits (matrix_coder LEN_BITS)
                    std::vector<uint8_t> b = encode_matrix(x.data(), x.size(), t->width, GLOBAL);
                    t->matrix = {"matrix", 0, t->n, t->width, std::string(b.begin(), b.end()), since(t1)};
                    t->matrix_ok = true;
                } catch (const std::exception& e) {
                    t->matrix_error = std::string("gaintrie: ") + e.what();
                }
                done(*t);
            });
            for (size_t k = 0; k < nbins; k++) {
                pool.submit([this, t, k, &path, at] {
                    auto t1 = Clock::now();
                    uint64_t s = k * bin_size_, m = std::min(bin_size_, t->n - s);
                    std::vector<uint64_t> x = read_values(path, at + s * (t->width / 8), m, t->width);
                    std::vector<uint8_t> b = encode_pair(x.data(), m, t->width);
                    t->bins[k] = {"pair", s, m, t->width, std::string(b.begin(), b.end()), since(t1)};
                    done(*t);
                });
            }
        }
        pool.wait();
    }

    void finish(const std::string& index) {
        out_.write(index.data(), std::streamsize(index.size()));
        std::string n = le64_bytes(index.size());
        out_.write(n.data(), 8);
        out_.close();
        if (!out_) throw std::runtime_error("archive write failed");
    }

    std::vector<UnitRow> unit_rows;
    std::vector<TensorRow> tensor_rows;

private:
    void done(Tensor& t) {
        if (--t.pending == 0) finalize(t);
    }

    void finalize(Tensor& t) {
        Json rec = Json::object();
        rec.set("name", t.name);
        rec.set("dtype", t.info.at("dtype"));
        rec.set("shape", t.info.at("shape"));
        Json offs = Json::array();
        offs.push(Json(t.start));
        offs.push(Json(t.end));
        rec.set("data_offsets", offs);
        TensorRow row;
        std::vector<Unit>* units;
        std::vector<Unit> single;
        if (!t.width) {
            single.push_back(std::move(t.raw));
            units = &single;
        } else {
            row.has_bins = true;
            for (const Unit& u : t.bins) row.bins_bytes += u.blob.size(), row.bins_s += u.enc_s;
            if (t.matrix_ok) {
                row.has_matrix = true;
                row.matrix_bytes = t.matrix.blob.size();
                row.matrix_s = t.matrix.enc_s;
            } else {
                rec.set("matrix_error", t.matrix_error);
            }
            bool matrix = t.matrix_ok && row.matrix_bytes < row.bins_bytes;
            rec.set("choice", matrix ? "matrix" : "bins");
            if (matrix) {
                single.push_back(std::move(t.matrix));
                units = &single;
            } else {
                units = &t.bins;
            }
        }
        auto [kind, layer] = tensor_type(t.name);
        std::lock_guard<std::mutex> l(m_);
        Json ulist = Json::array();
        for (Unit& u : *units) {
            out_.write(u.blob.data(), std::streamsize(u.blob.size()));
            Json j = Json::object();
            j.set("kind", u.kind);
            j.set("start", Json(u.start));
            j.set("values", Json(u.values));
            j.set("width", u.width);
            j.set("offset", Json(offset_));
            j.set("length", Json(u.blob.size()));
            ulist.push(std::move(j));
            offset_ += u.blob.size();
            row.stored_bytes += u.blob.size();
            uint64_t raw_bytes = u.values * u.width / 8;
            unit_rows.push_back({t.name, kind, u.kind, layer, u.start, u.values, raw_bytes, u.blob.size(), u.enc_s});
        }
        if (!out_) throw std::runtime_error("archive write failed");
        rec.set("units", std::move(ulist));
        (*shards_)[t.shard].records.push_back(std::move(rec));
        row.tensor = t.name;
        row.type = kind;
        row.layer = layer;
        row.dtype = t.info.at("dtype").as_str();
        row.raw_bytes = t.end - t.start;
        int w = codec_width(row.dtype);
        row.values = row.raw_bytes / std::max(1, (w ? w : 8) / 8);
        row.choice = t.width ? (row.has_matrix && row.matrix_bytes < row.bins_bytes ? "matrix" : "bins") : "raw";
        row.note = t.matrix_error;
        tensor_rows.push_back(row);
        t.bins = {};
        t.matrix = t.raw = {};
        size_t i = tensor_rows.size();
        if (i % 25 == 0 || i == total_) {
            uint64_t raw = 0, stored = 0;
            for (const TensorRow& r : tensor_rows) raw += r.raw_bytes, stored += r.stored_bytes;
            std::printf("  compressed %zu/%zu tensors, %.2f GB -> %.2f GB (%.4f), %.0fs\n", i, total_, raw / 1e9,
                        stored / 1e9, double(stored) / double(std::max<uint64_t>(raw, 1)), since(t0_));
            std::fflush(stdout);
        }
    }

    std::ofstream out_;
    uint64_t bin_size_;
    Clock::time_point t0_;
    uint64_t offset_ = MAGIC_LEN;
    size_t total_ = 0;
    std::vector<Shard>* shards_ = nullptr;
    std::mutex m_;
};

Json read_index(const fs::path& archive) {
    std::ifstream f(archive, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + archive.string());
    char magic[MAGIC_LEN];
    f.read(magic, MAGIC_LEN);
    if (f.gcount() != std::streamsize(MAGIC_LEN) || std::string(magic, MAGIC_LEN) != MAGIC)
        throw std::runtime_error("not a CLAR1 archive");
    uint64_t size = fs::file_size(archive);
    if (size < MAGIC_LEN + 8) throw std::runtime_error("archive too short");
    uint8_t nb[8];
    f.seekg(std::streamoff(size - 8));
    f.read(reinterpret_cast<char*>(nb), 8);
    uint64_t n = le64(nb);
    if (n > size - 8 - MAGIC_LEN) throw std::runtime_error("bad index length");
    return json_parse(read_at(archive, size - 8 - n, n));
}

}  // namespace

bool DecompressResult::lossless() const {
    for (const ShardCheck& c : checks)
        if (!c.matches_archive || c.matches_reference == 0) return false;
    return true;
}

CompressResult pipeline_compress(const std::string& shard_dir, const std::string& archive, const std::string& run_dir,
                                 uint64_t bin_size, int workers) {
    if (bin_size == 0) throw std::runtime_error("bin size must be >= 1");
    fs::create_directories(run_dir);
    if (fs::path(archive).has_parent_path()) fs::create_directories(fs::path(archive).parent_path());
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(shard_dir))
        if (e.is_regular_file() && e.path().extension() == ".safetensors") paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    auto t0 = Clock::now();

    std::vector<Shard> shards(paths.size());
    std::vector<std::unique_ptr<Tensor>> tensors;
    CompressResult res;
    for (size_t s = 0; s < paths.size(); s++) {
        Shard& sh = shards[s];
        sh.path = paths[s];
        sh.size = fs::file_size(sh.path);
        res.input_bytes += sh.size;
        if (sh.size < 8) throw std::runtime_error("not a safetensors file: " + sh.path.string());
        std::string nb = read_at(sh.path, 0, 8);
        uint64_t n = le64(reinterpret_cast<const uint8_t*>(nb.data()));
        if (n > sh.size - 8) throw std::runtime_error("bad header length: " + sh.path.string());
        sh.header = read_at(sh.path, 8, n);
        sh.base = 8 + n;
        Json header = json_parse(sh.header);
        for (auto& [name, info] : header.obj) {
            if (name == "__metadata__") continue;
            auto t = std::make_unique<Tensor>();
            t->shard = s;
            t->name = name;
            t->info = info;
            const Json& offs = info.at("data_offsets");
            if (offs.arr.size() != 2) throw std::runtime_error("bad data_offsets: " + name);
            t->start = uint64_t(offs.arr[0].as_int());
            t->end = uint64_t(offs.arr[1].as_int());
            if (t->end < t->start || sh.base + t->end > sh.size) throw std::runtime_error("bad data_offsets: " + name);
            int w = codec_width(info.at("dtype").as_str());
            if (w && t->end > t->start && (t->end - t->start) % (w / 8) == 0) {
                t->width = w;
                t->n = (t->end - t->start) / (w / 8);
            }
            tensors.push_back(std::move(t));
        }
    }

    ThreadPool hasher(int(std::min<size_t>(std::max<size_t>(shards.size(), 1), 4)));
    for (Shard& sh : shards) hasher.submit([&sh] { sh.sha256 = sha256_file(sh.path.string()); });

    Compressor comp(archive, bin_size, t0);
    comp.run(shards, tensors, workers);
    hasher.wait();
    Json index = Json::object();
    index.set("bin_size", Json(bin_size));
    Json& list = index.set("shards", Json::array());
    for (Shard& sh : shards) {
        std::stable_sort(sh.records.begin(), sh.records.end(), [](const Json& a, const Json& b) {
            return a.at("data_offsets").arr[0].i < b.at("data_offsets").arr[0].i;
        });
        Json e = Json::object();
        e.set("file", sh.path.filename().string());
        e.set("size", Json(sh.size));
        e.set("header", base64_encode(sh.header));
        Json recs = Json::array();
        recs.arr = std::move(sh.records);
        e.set("tensors", std::move(recs));
        e.set("sha256", sh.sha256);
        list.push(std::move(e));
    }
    comp.finish(json_dump(index));
    res.compress_s = since(t0);
    res.archive_bytes = fs::file_size(archive);
    res.units = std::move(comp.unit_rows);
    res.tensors = std::move(comp.tensor_rows);

    std::vector<std::vector<std::string>> rows;
    for (const UnitRow& u : res.units) rows.push_back(unit_csv(u));
    write_csv(fs::path(run_dir) / "compress.csv",
              {"tensor", "type", "layer", "kind", "start", "values", "raw_bytes", "comp_bytes", "bits_per_value",
               "ratio", "enc_s"},
              rows);
    std::vector<TensorRow> sorted = res.tensors;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const TensorRow& a, const TensorRow& b) { return a.tensor < b.tensor; });
    rows.clear();
    for (const TensorRow& r : sorted) rows.push_back(tensor_csv(r));
    write_csv(fs::path(run_dir) / "tensors.csv",
              {"tensor", "type", "layer", "dtype", "values", "raw_bytes", "choice", "bins_bytes", "bins_s",
               "matrix_bytes", "matrix_s", "stored_bytes", "note"},
              rows);
    return res;
}

DecompressResult pipeline_decompress(const std::string& archive, const std::string& out_dir,
                                     const std::string& run_dir, int workers, const std::string& reference) {
    fs::create_directories(out_dir);
    fs::create_directories(run_dir);
    auto t0 = Clock::now();
    Json index = read_index(archive);
    struct Job {
        fs::path path;
        uint64_t base;
        const std::string* name;
        const Json* unit;
    };
    std::vector<Job> jobs;
    for (const Json& entry : index.at("shards").arr) {
        fs::path path = fs::path(out_dir) / entry.at("file").as_str();
        std::string header = base64_decode(entry.at("header").as_str());
        {
            std::ofstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("cannot write " + path.string());
            f << le64_bytes(header.size()) << header;
            if (!f) throw std::runtime_error("write failed: " + path.string());
        }
        fs::resize_file(path, uint64_t(entry.at("size").as_int()));
        uint64_t base = 8 + header.size();
        for (const Json& rec : entry.at("tensors").arr)
            for (const Json& u : rec.at("units").arr)
                jobs.push_back({path, base + uint64_t(rec.at("data_offsets").arr.at(0).as_int()), &rec.at("name").s, &u});
    }
    std::stable_sort(jobs.begin(), jobs.end(),
                     [](const Job& a, const Job& b) { return a.unit->at("length").i > b.unit->at("length").i; });

    std::vector<std::vector<std::string>> rows;
    std::mutex m;
    {
        ThreadPool pool(workers);
        for (const Job& j : jobs) {
            pool.submit([&, j] {
                auto t = Clock::now();
                const Json& u = *j.unit;
                const std::string& kind = u.at("kind").as_str();
                uint64_t start = uint64_t(u.at("start").as_int()), values = uint64_t(u.at("values").as_int());
                int width = int(u.at("width").as_int());
                std::string blob = read_at(archive, uint64_t(u.at("offset").as_int()), uint64_t(u.at("length").as_int()));
                std::string data;
                if (kind == "raw") {
                    data = std::move(blob);
                } else {
                    const uint8_t* p = reinterpret_cast<const uint8_t*>(blob.data());
                    if (kind == "pair") data = decode_pair_le(p, blob.size(), width / 8);
                    else if (kind == "matrix") data = decode_matrix_le(p, blob.size(), width / 8);
                    else throw std::runtime_error(*j.name + ": unknown unit kind " + kind);
                }
                if (data.size() != values * width / 8) throw std::runtime_error(*j.name + ": wrong decoded length");
                {
                    std::fstream f(j.path, std::ios::binary | std::ios::in | std::ios::out);
                    f.seekp(std::streamoff(j.base + start * width / 8));
                    f.write(data.data(), std::streamsize(data.size()));
                    if (!f) throw std::runtime_error("write failed: " + j.path.string());
                }
                double dt = since(t);
                std::lock_guard<std::mutex> l(m);
                rows.push_back({*j.name, kind, std::to_string(start), std::to_string(values),
                                std::to_string(u.at("length").i), py_float(py_round(dt, 4))});
                if (rows.size() % 500 == 0 || rows.size() == jobs.size()) {
                    std::printf("  decoded %zu/%zu units, %.0fs\n", rows.size(), jobs.size(), since(t0));
                    std::fflush(stdout);
                }
            });
        }
        pool.wait();
    }
    DecompressResult res;
    res.decompress_s = since(t0);

    t0 = Clock::now();
    std::map<std::string, std::string> expected;
    if (!reference.empty() && fs::exists(fs::path(reference) / "sha256.txt")) {
        std::ifstream f(fs::path(reference) / "sha256.txt");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ls(line);
            std::string digest, file;
            if (ls >> digest >> file) expected[file] = digest;
        }
    }
    const auto& entries = index.at("shards").arr;
    res.checks.resize(entries.size());
    {
        ThreadPool pool(int(std::min<size_t>(std::max<size_t>(entries.size(), 1), size_t(std::max(workers, 1)))));
        for (size_t i = 0; i < entries.size(); i++) {
            pool.submit([&, i] {
                ShardCheck& c = res.checks[i];
                c.file = entries[i].at("file").as_str();
                c.sha256 = sha256_file((fs::path(out_dir) / c.file).string());
                c.matches_archive = c.sha256 == entries[i].at("sha256").as_str();
                auto it = expected.find(c.file);
                if (it != expected.end()) c.matches_reference = it->second == c.sha256;
            });
        }
        pool.wait();
    }
    res.verify_s = since(t0);
    write_csv(fs::path(run_dir) / "decompress.csv", {"tensor", "kind", "start", "values", "comp_bytes", "dec_s"},
              rows);
    return res;
}

void print_checks(const DecompressResult& dec) {
    for (const ShardCheck& c : dec.checks)
        std::printf("  %s: archive sha256 %s, reference %s\n", c.file.c_str(), c.matches_archive ? "OK" : "MISMATCH",
                    c.matches_reference < 0 ? "-" : c.matches_reference ? "OK" : "MISMATCH");
    std::printf("%s\n", dec.lossless() ? "LOSSLESS" : "NOT LOSSLESS");
}

void pipeline_summarize(const CompressResult& comp, const DecompressResult* dec, const std::string& run_dir) {
    const auto& units = comp.units;
    const auto& tensors = comp.tensors;
    struct Group {
        uint64_t n = 0, values = 0, raw = 0, stored = 0, bins = 0;
    };
    std::vector<std::pair<std::string, Group>> groups;
    for (const TensorRow& r : tensors) {
        auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == r.type; });
        if (it == groups.end()) it = groups.insert(groups.end(), {r.type, Group{}});
        Group& g = it->second;
        g.n++, g.values += r.values, g.raw += r.raw_bytes, g.stored += r.stored_bytes, g.bins += r.choice == "bins";
    }
    std::stable_sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) { return a.second.raw > b.second.raw; });
    auto div = [](double a, double b) { return b ? a / b : 0.0; };
    auto thousands = [](uint64_t v) {
        std::string s = std::to_string(v);
        for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(size_t(i), ",");
        return s;
    };
    std::printf("==Per tensor type== (stored = smaller candidate)\n");
    std::printf("%-34s%8s%15s%9s%11s%8s%8s%10s\n", "type", "tensors", "values", "raw GB", "stored GB", "ratio",
                "bits/v", "bins won");
    for (const auto& [kind, g] : groups)
        std::printf("%-34s%8llu%15s%9.3f%11.3f%8.4f%8.3f%7llu/%llu\n", kind.c_str(), (unsigned long long)g.n,
                    thousands(g.values).c_str(), g.raw / 1e9, g.stored / 1e9, div(g.stored, g.raw),
                    div(g.stored * 8.0, g.values), (unsigned long long)g.bins, (unsigned long long)g.n);
    std::map<int64_t, std::pair<uint64_t, uint64_t>> layers;
    for (const TensorRow& r : tensors)
        if (r.layer >= 0) layers[r.layer].first += r.raw_bytes, layers[r.layer].second += r.stored_bytes;
    std::string line = "==Per layer== ratio: ";
    bool first = true;
    for (const auto& [l, rs] : layers) {
        line += fmt("%s%lld:%.4f", first ? "" : "  ", (long long)l, div(rs.second, rs.first));
        first = false;
    }
    std::printf("%s\n", line.c_str());
    std::vector<double> bpv;
    size_t nbins = 0, nmat = 0, nraw = 0;
    for (const UnitRow& u : units) {
        nbins += u.kind == "pair", nmat += u.kind == "matrix", nraw += u.kind == "raw";
        if (u.kind != "raw") bpv.push_back(u.values ? py_round(u.comp_bytes * 8.0 / double(u.values), 4) : 0.0);
    }
    std::sort(bpv.begin(), bpv.end());
    std::printf("==Per unit== %zu units (%zu bins, %zu matrices, %zu raw)", units.size(), nbins, nmat, nraw);
    if (bpv.empty()) {
        std::printf("\n");
    } else {
        size_t h = bpv.size() / 2;
        double median = bpv.size() % 2 ? bpv[h] : (bpv[h - 1] + bpv[h]) / 2;
        std::printf("; bits/value min %.3f, median %.3f, max %.3f\n", bpv.front(), median, bpv.back());
    }

    uint64_t tensor_bytes = 0, stored = 0, cand_bins = 0, cand_matrix = 0, n_values = 0;
    for (const TensorRow& r : tensors) {
        tensor_bytes += r.raw_bytes;
        stored += r.stored_bytes;
        cand_bins += r.bins_bytes ? r.bins_bytes : r.stored_bytes;
        cand_matrix += r.matrix_bytes ? r.matrix_bytes : r.stored_bytes;
        n_values += r.values;
    }
    Json total = Json::object();
    total.set("input_bytes", Json(comp.input_bytes));
    total.set("archive_bytes", Json(comp.archive_bytes));
    total.set("ratio", div(double(comp.archive_bytes), double(comp.input_bytes)));
    total.set("tensor_bytes", Json(tensor_bytes));
    total.set("tensor_stored_bytes", Json(stored));
    total.set("candidate_bins_bytes", Json(cand_bins));
    total.set("candidate_matrix_bytes", Json(cand_matrix));
    total.set("compress_s", py_round(comp.compress_s, 1));
    if (dec) {
        total.set("decompress_s", py_round(dec->decompress_s, 1));
        total.set("verify_s", py_round(dec->verify_s, 1));
        total.set("lossless", dec->lossless());
        Json checks = Json::array();
        for (const ShardCheck& c : dec->checks) {
            Json j = Json::object();
            j.set("file", c.file);
            j.set("sha256", c.sha256);
            j.set("matches_archive", c.matches_archive);
            j.set("matches_reference", c.matches_reference < 0 ? Json(nullptr) : Json(c.matches_reference == 1));
            checks.push(std::move(j));
        }
        total.set("checks", std::move(checks));
    }
    std::printf("==Total==\n");
    std::printf("input %.3f GB -> archive %.3f GB, ratio %.4f (%.3f bits per value incl. headers + index, raw %.3f)\n",
                comp.input_bytes / 1e9, comp.archive_bytes / 1e9, total.at("ratio").f,
                div(comp.archive_bytes * 8.0, n_values), div(comp.input_bytes * 8.0, n_values));
    std::printf("only bins: %.3f GB, only matrix: %.3f GB, best per tensor: %.3f GB\n", cand_bins / 1e9,
                cand_matrix / 1e9, stored / 1e9);
    std::printf("time: compress %ss", py_float(total.at("compress_s").f).c_str());
    if (dec)
        std::printf(", decompress %ss, verify %ss", py_float(total.at("decompress_s").f).c_str(),
                    py_float(total.at("verify_s").f).c_str());
    std::printf("\n");
    if (dec) print_checks(*dec);
    std::fflush(stdout);
    std::ofstream f(fs::path(run_dir) / "summary.json", std::ios::binary);
    f << json_dump(total, 1);
}

bool pipeline_roundtrip(const std::string& shard_dir, const std::string& run_dir, uint64_t bin_size, int workers) {
    std::string archive = (fs::path(run_dir) / "model.clar").string();
    std::string restored = (fs::path(run_dir) / "restored").string();
    std::printf("######## compress %s -> %s\n", shard_dir.c_str(), archive.c_str());
    std::fflush(stdout);
    CompressResult comp = pipeline_compress(shard_dir, archive, run_dir, bin_size, workers);
    std::printf("######## decompress %s -> %s\n", archive.c_str(), restored.c_str());
    std::fflush(stdout);
    DecompressResult dec = pipeline_decompress(archive, restored, run_dir, workers, shard_dir);
    pipeline_summarize(comp, &dec, run_dir);
    return dec.lossless();
}

}  // namespace cle
