#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cle {

// gaintrie/pipeline.py: safetensors shards <-> one CLAR1 archive, same format and metrics files

struct UnitRow {
    std::string tensor, type, kind;
    int64_t layer = -1;  // -1: not a model.layers.N tensor
    uint64_t start = 0, values = 0, raw_bytes = 0, comp_bytes = 0;
    double enc_s = 0;
};

struct TensorRow {
    std::string tensor, type, dtype, choice, note;
    int64_t layer = -1;
    uint64_t values = 0, raw_bytes = 0, bins_bytes = 0, matrix_bytes = 0, stored_bytes = 0;
    bool has_bins = false, has_matrix = false;
    double bins_s = 0, matrix_s = 0;
};

struct ShardCheck {
    std::string file, sha256;
    bool matches_archive = false;
    int matches_reference = -1;  // -1: no reference hash
};

struct CompressResult {
    double compress_s = 0;
    uint64_t input_bytes = 0, archive_bytes = 0;
    std::vector<UnitRow> units;
    std::vector<TensorRow> tensors;
};

struct DecompressResult {
    double decompress_s = 0, verify_s = 0;
    std::vector<ShardCheck> checks;
    bool lossless() const;
};

CompressResult pipeline_compress(const std::string& shard_dir, const std::string& archive, const std::string& run_dir,
                                 uint64_t bin_size = 1 << 20, int workers = 8);
// reference: model dir whose sha256.txt is checked too ("" = none)
DecompressResult pipeline_decompress(const std::string& archive, const std::string& out_dir,
                                     const std::string& run_dir, int workers = 8, const std::string& reference = "");
// prints the summaries, writes summary.json; dec may be null
void pipeline_summarize(const CompressResult& comp, const DecompressResult* dec, const std::string& run_dir);
void print_checks(const DecompressResult& dec);
bool pipeline_roundtrip(const std::string& shard_dir, const std::string& run_dir, uint64_t bin_size = 1 << 20,
                        int workers = 8);

}  // namespace cle
