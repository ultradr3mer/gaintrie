# gaintrie

Lossless compression for neural-network weights.

**Qwen3-8B (bf16): 16.38 GB → 10.84 GB (66.2 %), restored byte for byte** — the rebuilt
safetensors files match the SHA-256 hashes published on Hugging Face. Compressing the whole
model takes 26 s, decompressing 11 s (16-core desktop, `gaintrie` C++ tool, 8 workers).

## How it compares

All numbers on the same input: the 5 safetensors shards of
[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), 16.38 GB, 8.19 billion bf16 weights
(raw: 16 bits per weight). Smaller is better.

| method | size | of original | bits per weight | time |
|---|---:|---:|---:|---:|
| **gaintrie** | **10.84 GB** | **66.2 %** | **10.59** | 26 s |
| Huffman on the exponent bits only (the idea behind DFloat11 / ZipNN) ¹ | 10.92 GB | 66.7 % | 10.67 | – |
| bzip2 -9 | 11.36 GB | 69.4 % | 11.10 | 208 s |
| xz -6 | 11.62 GB | 70.9 % | 11.35 | 640 s |
| zstd -19 | 12.75 GB | 77.8 % | 12.45 | 640 s |
| zstd -3 | 12.86 GB | 78.5 % | 12.56 | 5 s |
| gzip -9 | 13.06 GB | 79.7 % | 12.76 | 78 s |
| *limit for coding each value on its own* ² | *10.82 GB* | *66.0 %* | *10.57* | |

- gaintrie is the smallest of all measured methods and **0.24 % above the limit** ² — the
  limit leaves out the table every real coder has to store, so it cannot be reached.
- General-purpose compressors lose 3–14 percentage points because they look for repeated
  byte sequences, which weights hardly contain.
- Published numbers for the specialised tools: [DFloat11](https://arxiv.org/abs/2504.11651)
  reports about 70 % (≈ 11 bits per weight), [ZipNN](https://arxiv.org/abs/2411.05239) about
  33 % savings on popular bf16 models.

¹ Computed with this repository's Huffman code per tensor: the 8 exponent bits Huffman-coded,
sign and mantissa stored as they are, plus a code table — not the DFloat11 or ZipNN software.
² Order-0 entropy of the 16-bit values, per tensor, without any table.
General-purpose compressors ran on the 5 files in parallel (zstd / xz / gzip with 4 threads each);
reproduce with `python research/compare_baselines.py`.

### Small tensors

Where the dictionary of values is a large part of the data, the trie pays off most
(Qwen3-8B, each tensor compressed on its own):

| tensor type | count × values | gaintrie | xz -9e | zstd -19 |
|---|---|---:|---:|---:|
| input_layernorm | 36 × 4096 | **7.77** | 8.59 | 9.66 |
| post_attention_layernorm | 36 × 4096 | **6.98** | 7.82 | 8.94 |
| q_norm | 36 × 128 | **12.02** | 16.68 | 13.49 |
| k_norm | 36 × 128 | **12.76** | 17.66 | 14.06 |
| model.norm | 1 × 4096 | **5.98** | 7.58 | 8.10 |

(bits per weight; gaintrie is 10–28 % smaller than xz -9e here)

## How it works

A weight tensor contains millions of values but only a few thousand *distinct* ones
(a 50-million-weight matrix of Qwen3-8B has about 5 000). gaintrie stores two things:

1. **The dictionary** — every distinct value once, as a binary tree over its bits (the *trie*).
2. **The stream** — for every weight, a short code saying which dictionary entry it is.
   Frequent values get short codes (Huffman coding).

The trie is where the method is its own. Instead of storing the bits of each value in a
fixed order, it repeatedly asks *"which bit tells the remaining values apart best?"*:

- **Split** — the node divides its values by the bit with the largest *entropy gain*,
  so every node picks its own bit, not the next one in line.
- **Strait** — a bit that is the same for all values below a node is written once, right
  there, instead of once per value.
- **Pair** — a node with exactly two values stores which bits differ plus one value; the other
  value follows by flipping those bits.
- **Leaf** — one value; only its bits not yet fixed above are stored.

### Example

Four 4-bit values with how often they occur: `A = 0110` ×5, `B = 0111` ×2, `C = 1110` ×1,
`D = 0100` ×1 (bit 0 is the leftmost). This is the tree the coder actually builds:

```text
A, B, C, D              bit 1 is 1 in all of them    → strait "bit 1 = 1", stored once
└─ split on bit 3
   ├─ 1: B              bits 0 and 2 still open      → leaf "01"
   └─ 0: A, C, D
      └─ split on bit 2
         ├─ 0: D        bit 0 still open             → leaf "0"
         └─ 1: A, C     only bit 0 is open           → split on bit 0 (position costs 0 bits)
            ├─ 1: C     nothing open                 → leaf, 0 bits
            └─ 0: A     nothing open                 → leaf, 0 bits
```

Every bit of every value appears exactly once in the tree, and the shared bit 1 only once for
all four. The stream then says which leaf each weight is:

| | A ×5 | B ×2 | C ×1 | D ×1 | stream for all 9 weights |
|---|---|---|---|---|---:|
| path through the tree (code bits = value bits) | `010` | `1` | `011` | `00` | 22 bits |
| Huffman codes by count | `0` | `10` | `111` | `110` | **15 bits** |

Here the encoder picks Huffman. A **pair** node would replace a two-value subtree such as
A, C when that is cheaper — not here, where the split on the last open bit is already free.

### What gets written

- Positions are numbers among the bits still open at that node, so they shrink as the tree
  goes down (one open bit costs nothing).
- The stream is either Huffman codes over the leaves or the path through the tree itself
  (then a value's code *is* its bits); the encoder takes whichever is smaller.
- Large tensors are either cut into bins of 1 M values (each with its own tree and table) or
  stored with one tree and one Huffman table for the whole matrix — the pipeline tries both
  per tensor and keeps the smaller.
- Exact formats: module docstrings of `gaintrie/pair_writer.py`, `gaintrie/matrix_coder.py`
  and `gaintrie/pipeline.py`.

## Results by tensor type (Qwen3-8B)

| tensor type | count | of original | bits per weight |
|---|---:|---:|---:|
| mlp.down_proj / up_proj / gate_proj | 3 × 36 | 66.2–66.4 % | 10.60–10.63 |
| embed_tokens, lm_head | 1 each | 66.0 % | 10.56 |
| self_attn.q_proj / o_proj | 36 each | 65.9–66.1 % | 10.54–10.58 |
| self_attn.k_proj / v_proj | 36 each | 66.1 % | 10.58–10.59 |
| input_layernorm / post_attention_layernorm | 36 each | 48.6 % / 43.6 % | 7.77 / 6.98 |
| q_norm / k_norm | 36 each | 75.1 % / 79.7 % | 12.02 / 12.76 |
| **whole archive** | 399 tensors | **66.2 %** | **10.59** |

The pipeline logs every bin (`compress.csv`), both candidates per tensor (`tensors.csv`), decode
times (`decompress.csv`) and the totals with the SHA-256 checks (`summary.json`).

## Compress a model

Only a C++17 compiler and CMake are needed — no Python:

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build -j
```

Compress, decompress and verify a model (a directory of `.safetensors` files, optionally with
a `sha256.txt` of reference hashes):

```bash
cpp/build/gaintrie roundtrip <model dir> <run dir>
```

The run directory then contains `model.clar` (the archive), `restored/` (the rebuilt files)
and the metrics. The two halves on their own:

```bash
cpp/build/gaintrie compress <model dir> model.clar [--workers 8] [--bin-size 1048576] [--run-dir D]
cpp/build/gaintrie decompress model.clar <out dir> [--reference <model dir>] [--workers 8] [--run-dir D]
```

`decompress` checks every rebuilt file against the SHA-256 stored in the archive (and against
`<model dir>/sha256.txt` with `--reference`) and exits with code 3 if one differs.
Single tensors as raw little-endian values:

```bash
cpp/build/gaintrie encode --format matrix --width 16 --mode global --stats weights.bin out.cle
cpp/build/gaintrie decode --format matrix out.cle weights.bin
```

## Python reference, research and tests

Requirements: Python ≥ 3.14 (clarautils uses lazy annotations), numpy, clarautils, and the C++
build above (the Python pipeline and the compatibility tests call `cpp/build/gaintrie`).

```bash
pip install numpy "clarautils @ git+https://github.com/ultradr3mer/BitFlagArray"
python gaintrie/pipeline.py roundtrip <model dir> <run dir>
```

Both pipelines write the same archive format and metrics files; archives from one decompress
with the other. Tests (each Python file also runs without pytest):

```bash
cpp/build/test_roundtrip
cpp/build/test_pipeline
python tests/test_pair_writer.py
python tests/test_matrix_coder.py
python tests/test_pipeline.py
python tests/test_cpp_compat.py
python tests/test_cpp_pipeline.py
```

## Repository layout

| path | content |
|---|---|
| `gaintrie/` | Python reference: trie (`GainCoder`), writers (`pair_writer`, `matrix_coder`), `pipeline`, `split_safetensors` |
| `cpp/` | C++17 implementation: codec library, `gaintrie` CLI with the whole pipeline (compress / decompress / roundtrip), tests |
| `research/` | experiments behind the design: flip masks, strait rules, bin sizes, bit orders, leaf statistics, baselines |
| `tests/` | Python round-trip tests and C++ ↔ Python compatibility tests |

## Limits

- One matrix stream holds at most 2³² bits; larger tensors (the 622 M-weight embeddings) are
  stored as 1 M-value bins instead.
- The Python version is the readable reference and slow; use the C++ tool for real models.
- The Python trie builder uses [clarautils](https://github.com/ultradr3mer/BitFlagArray).

## License

MIT
