"""One container per matrix with a global Huffman table (written + decoded).

  header   N:32  width-1:6  mode:1  bin size:32
  segment  bit length:32, then the bits; segments follow each other:
    global   PairWriter tree over all distinct values of the matrix (the dictionary,
             leaf order = order the decoder rebuilds them in) + one Huffman code
             length per leaf (pair_writer.put_code_lengths, by global count)
    GLOBAL   stream: every item's canonical code from the global lengths
    HYBRID   per bin: the bin's own PairWriter tree (which values occur), then its
             stream; the bin's codes are Huffman over weights 2^(max - global length)
             of the bin's leaves, so the bin carries no code table of its own
The stream is written / read vectorized (numpy), the trees with pair_writer.
"""
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clarautils import BitWriter, BitReader
from gaintrie.GainCoder import GainCoder
from gaintrie.code_utils import huffman_lengths, canonical_codes
from gaintrie.pair_writer import (PairWriter, write_tree, read_tree, put_code_lengths, read_code_lengths,
                                  code_length_bits, RAW, COMB)
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import put_int, bits_written

GLOBAL, HYBRID = 0, 1
LEN_BITS = 32
CHUNK = 1 << 24  # bit positions per decode chunk


def writer_bits(bw: BitWriter) -> np.ndarray:
    return np.unpackbits(bw.get_bytes())[:bits_written(bw)]


def int_bits(value: int, length: int) -> np.ndarray:
    return np.array([(value >> (length - 1 - i)) & 1 for i in range(length)], dtype=np.uint8)


def bits_int(bits) -> int:
    return int("".join(map(str, bits.tolist())) or "0", 2)


def tree_segment(values, counts, width):
    """smaller PairWriter tree (raw / comb pair masks) -> (BitWriter, leaf values in decoder order)"""
    coder = GainCoder(values, counts, width)
    best = None
    for mode in (RAW, COMB):
        bw = BitWriter()
        _, table = write_tree(PairWriter(coder, mode), bw)
        if best is None or bits_written(bw) < bits_written(best[0]):
            best = (bw, list(table))
    return best


def code_values(lengths) -> np.ndarray:
    return np.array([int(c, 2) if c else 0 for c in canonical_codes(lengths)], dtype=np.uint64)


def stream_bits(symbols: np.ndarray, lengths) -> np.ndarray:
    """canonical Huffman codes of `symbols` (indices into lengths) as a bit array"""
    lens = np.asarray(lengths, dtype=np.int64)[symbols]
    vals = code_values(lengths)[symbols]
    starts = np.cumsum(lens) - lens
    out = np.zeros(int(lens.sum()), dtype=np.uint8)
    for j in range(int(lens.max()) if len(out) else 0):
        m = lens > j
        out[starts[m] + j] = (vals[m] >> (lens[m] - 1 - j).astype(np.uint64)) & np.uint64(1)
    return out


def decode_stream(bits: np.ndarray, n: int, lengths) -> np.ndarray:
    """n symbols from a canonical Huffman bit array: per chunk the code length and symbol
    starting at every bit position are computed vectorized, then only the starts are followed"""
    max_len = max(lengths) if lengths else 0
    if n == 0 or max_len == 0:
        return np.zeros(n, dtype=np.int64)
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    vals = code_values(lengths)
    sorted_syms = np.array(order, dtype=np.int64)
    by_len = {}
    for idx, s in enumerate(order):
        first, cnt, first_idx = by_len.get(lengths[s], (int(vals[s]), 0, idx))
        by_len[lengths[s]] = (first, cnt + 1, first_idx)
    padded = np.concatenate((bits, np.zeros(max_len, dtype=np.uint8))).astype(np.uint64)
    out, k, pos = [], 0, 0
    while k < n:
        c0, c1 = pos, min(pos + CHUNK, len(bits))
        win = np.zeros(c1 - c0, dtype=np.uint64)
        for j in range(max_len):
            win = (win << np.uint64(1)) | padded[c0 + j:c1 + j]
        len_at = np.zeros(c1 - c0, dtype=np.int64)
        sym_at = np.zeros(c1 - c0, dtype=np.int64)
        for l, (first, cnt, first_idx) in sorted(by_len.items()):
            v = win >> np.uint64(max_len - l)
            hit = (len_at == 0) & (v >= np.uint64(first)) & (v < np.uint64(first + cnt))
            len_at[hit] = l
            sym_at[hit] = sorted_syms[first_idx + (v[hit] - np.uint64(first)).astype(np.int64)]
        la, sa = len_at.tolist(), sym_at.tolist()
        chunk = []
        while k < n and pos < c1:
            chunk.append(sa[pos - c0])
            pos += la[pos - c0]
            k += 1
        out.append(np.array(chunk, dtype=np.int64))
    return np.concatenate(out)


def leaf_symbols(x, leaves) -> np.ndarray:
    """index of every item's value in `leaves`"""
    leaves = np.asarray(leaves, dtype=np.uint64)
    order = np.argsort(leaves)
    return order[np.searchsorted(leaves[order], np.asarray(x, dtype=np.uint64))]


def bin_lengths(leaves, global_len: dict, max_len: int) -> list:
    return huffman_lengths([1 << (max_len - global_len[v]) for v in leaves])


def encode_matrix(x, width: int, mode: int, bin_size: int = 1 << 20):
    x = np.asarray(x)
    n = len(x)
    assert n < 1 << 32 and bin_size < 1 << 32
    stats, parts = Counter(), []

    def segment(name, bits):
        assert len(bits) < 1 << LEN_BITS
        parts.extend((int_bits(len(bits), LEN_BITS), bits))
        stats["framing"] += LEN_BITS
        stats[name] += len(bits)

    head = BitWriter()
    put_int(head, n, 32)
    put_int(head, width - 1, 6)
    put_int(head, mode, 1)
    put_int(head, bin_size, 32)
    parts.append(writer_bits(head))
    stats["header"] += 71

    values, counts = np.unique(x, return_counts=True)
    bw, leaves = tree_segment(values, counts, width)
    tree_bits = bits_written(bw)
    count_of = dict(zip(values.tolist(), counts.tolist()))
    lengths = huffman_lengths([count_of[v] for v in leaves])
    put_code_lengths(bw, lengths)
    segment("global_tree", writer_bits(bw))
    stats["global_tree"] -= code_length_bits(lengths)
    stats["global_lengths"] += code_length_bits(lengths)
    assert stats["global_tree"] == tree_bits

    if mode == GLOBAL:
        segment("stream", stream_bits(leaf_symbols(x, leaves), lengths))
    else:
        global_len, max_len = dict(zip(leaves, lengths)), max(lengths)
        for start in range(0, n, bin_size):
            b = x[start:start + bin_size]
            bv, bc = np.unique(b, return_counts=True)
            bbw, bleaves = tree_segment(bv, bc, width)
            segment("bin_trees", writer_bits(bbw))
            segment("stream", stream_bits(leaf_symbols(b, bleaves), bin_lengths(bleaves, global_len, max_len)))
    bits = np.concatenate(parts)
    assert len(bits) == sum(stats.values())
    return np.packbits(bits), stats


def decode_matrix(data) -> np.ndarray:
    bits = np.unpackbits(np.asarray(data, dtype=np.uint8))
    n, width, mode, bin_size = (bits_int(bits[0:32]), bits_int(bits[32:38]) + 1,
                                bits_int(bits[38:39]), bits_int(bits[39:71]))
    pos = 71

    def segment():
        nonlocal pos
        length = bits_int(bits[pos:pos + LEN_BITS])
        pos += LEN_BITS + length
        return bits[pos - length:pos]

    br = BitReader(np.packbits(segment()))
    leaves = list(read_tree(br).values())
    lengths = read_code_lengths(br, len(leaves))
    if mode == GLOBAL:
        return np.asarray(leaves, dtype=np.uint64)[decode_stream(segment(), n, lengths)]
    global_len, max_len = dict(zip(leaves, lengths)), max(lengths)
    out = []
    for start in range(0, n, bin_size):
        bleaves = list(read_tree(BitReader(np.packbits(segment()))).values())
        m = min(bin_size, n - start)
        syms = decode_stream(segment(), m, bin_lengths(bleaves, global_len, max_len))
        out.append(np.asarray(bleaves, dtype=np.uint64)[syms])
    return np.concatenate(out)


def report(name, x, width, mode, bin_size=1 << 20):
    t = time.time()
    data, stats = encode_matrix(x, width, mode, bin_size)
    t_enc = time.time() - t
    t = time.time()
    assert np.array_equal(decode_matrix(data), np.asarray(x, dtype=np.uint64)), "round trip failed"
    t_dec = time.time() - t
    n = len(x)
    label = "global" if mode == GLOBAL else f"hybrid bins={bin_size}"
    parts = "  ".join(f"{k} {v / n:.4f}" for k, v in stats.items())
    print(f"{name:<16} {label:<20} {sum(stats.values()) / n:.4f} bits/value  ({parts})  "
          f"round trip OK, enc {t_enc:.0f}s dec {t_dec:.0f}s", flush=True)
    return sum(stats.values())


if __name__ == "__main__":
    for matrix, configs in (("self_attn.k_proj", ((GLOBAL, 0), (HYBRID, 1 << 20), (HYBRID, 1 << 18), (HYBRID, 1 << 16))),
                            ("mlp.gate_proj", ((GLOBAL, 0), (HYBRID, 1 << 22)))):
        x = np.fromfile(sandbox_path("bins", f"model.layers.0.{matrix}.weight.bin"), dtype=np.uint16)
        c = np.bincount(x)
        c = c[c > 0]
        print(f"######## {matrix} layer 0: {len(x):,} values, entropy {float((c * np.log2(len(x) / c)).sum()) / len(x):.4f}")
        for mode, size in configs:
            report(matrix, x, 16, mode, size or 1 << 20)
    print("END")
