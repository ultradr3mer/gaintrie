# gaintrie (C++17)

Port of `gaintrie/pair_writer.py` (single bin) and `gaintrie/matrix_coder.py` (matrix container), same file
formats — files are interchangeable with the Python reference (`tests/test_cpp_compat.py`).

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
cpp/build/test_roundtrip
```

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
