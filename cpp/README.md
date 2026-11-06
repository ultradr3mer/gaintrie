# gaintrie (C++17)

Port of `gaintrie/pair_writer.py` (single bin), `gaintrie/matrix_coder.py` (matrix container) and
`gaintrie/pipeline.py` (whole models), same file formats — archives and files are interchangeable with the
Python reference (`tests/test_cpp_compat.py`, `tests/test_cpp_pipeline.py`). Standard library only.

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
cpp/build/test_roundtrip && cpp/build/test_pipeline
```

Without CMake: `g++ -std=c++17 -O3 -DNDEBUG -ffp-contract=off -pthread -Icpp/src cpp/src/*.cpp -o cpp/build/gaintrie`.

## Models

```bash
cpp/build/gaintrie roundtrip <model dir> <run dir> [--bin-size N] [--workers N]
cpp/build/gaintrie compress <model dir> <archive> [--run-dir D] [--bin-size N] [--workers N]
cpp/build/gaintrie decompress <archive> <out dir> [--reference <model dir>] [--run-dir D] [--workers N]
```

Same semantics, archive (`CLAR1` | unit blobs | index JSON | index length) and metrics files (`compress.csv`,
`tensors.csv`, `decompress.csv`, `summary.json`) as `pipeline.py`. Differences: work is scheduled per unit
(each bin and each matrix candidate is its own task, largest tensors first) instead of per tensor, so `bins_s`
is the summed time of the bins; the shards are hashed in parallel (SHA-NI when the CPU has it); `decompress`
exits with 3 when a SHA-256 check fails. Defaults: `--bin-size 1048576`, `--workers 8`, run dir = archive
directory (compress) / out dir (decompress).

## Single tensors

```bash
cpp/build/gaintrie encode --format matrix --width 16 --mode hybrid --bin-size 1048576 --stats in.bin out.cle
cpp/build/gaintrie decode --format matrix out.cle back.bin
```

Input: little-endian unsigned values, element size = smallest of 1/2/4/8 bytes holding `--width` bits
(`--in-bytes` / `--out-bytes` override). Pair format options: `--mask auto|raw|comb`,
`--stream auto|tree|huffman`, `--pos auto|msb|order` (defaults = `pair_writer.best_writer`).

Trees match the Python GainCoder except for rare split ties: clarautils computes the entropies in float32 with
numpy's (SIMD, not correctly rounded) `log2`; the port rounds `log2` via double.

`bench_compare.py`: Python vs C++ time and bits/value on real bins.
