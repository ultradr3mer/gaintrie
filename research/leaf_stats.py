"""How often do leaves repeat, and what would Huffman save?

A) leaf values in the data: each unique value is one leaf, its count is how
   often its code appears in the stream. The tree codes ignore counts, a
   Huffman code over the leaves would not (+ a table of code lengths).
B) leaf contents in the tree: identical residual bit patterns (GainCoder
   leaves, PairWriter leaves, pair masks / values). The decoder knows len(rem),
   so the estimate uses one Huffman table per length (+ a dictionary of the
   unique patterns).
Estimates only, nothing is written.
"""
from collections import Counter
from math import log2
from pathlib import Path
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.GainCoder import GainCoder
from gaintrie.code_utils import huffman_lengths
from gaintrie.pair_writer import best_writer, PAIR
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import walk, kind, pos_bits, bit_of, encode as gain_encode, LEAF


def entropy_bits(counts) -> float:
    n = sum(counts)
    return sum(c * log2(n / c) for c in counts)


def count_buckets(counts) -> str:
    """values / items per occurrence bucket 1, 2, 3-4, 5-8, ..."""
    buckets = Counter()
    items = Counter()
    for c in counts:
        b = (c - 1).bit_length()
        buckets[b] += 1
        items[b] += c
    label = lambda b: "1" if b == 0 else f"{(1 << (b - 1)) + 1}-{1 << b}" if b > 1 else "2"
    n = sum(counts)
    return ", ".join(f"{label(b)}×: {buckets[b]} values / {items[b] / n:.0%} items" for b in sorted(buckets))


def stream_stats(coder: GainCoder, x):
    values, counts = coder.values.tolist(), coder.counts.tolist()
    n, u = len(x), len(values)
    tree_stream = sum(len(coder.codes[v]) * c for v, c in zip(values, counts))
    lengths = huffman_lengths(counts)
    huff = sum(l * c for l, c in zip(lengths, counts))
    table = u * pos_bits(max(lengths) + 1)
    top = sorted(counts, reverse=True)
    print("==A) leaf values in the data (stream)==")
    print(f"items {n}, unique values (leaves) {u}, avg {n / u:.2f} items per leaf, max {top[0]}")
    print("occurrences:", count_buckets(counts))
    print(f"top 10 values cover {sum(top[:10]) / n:.0%} of the items, top 100 {sum(top[:100]) / n:.0%}")
    print(f"stream now (tree codes): {tree_stream} bits ({tree_stream / n:.3f}/item)")
    print(f"huffman over leaves:     {huff} bits ({huff / n:.3f}/item) + length table {u}×{pos_bits(max(lengths) + 1)}"
          f" = {table} -> {huff + table} ({huff + table - tree_stream:+d})")
    print(f"entropy bound:           {entropy_bits(counts):.0f} bits ({entropy_bits(counts) / n:.3f}/item)")


def pattern_stats(title: str, symbols: list):
    """symbols: bit tuples; one Huffman table + dictionary per length"""
    if not symbols:
        print(f"  {title}: none")
        return
    by_len = {}
    for s in symbols:
        by_len.setdefault(len(s), Counter())[s] += 1
    inline = sum(len(s) for s in symbols)
    huff = table = 0
    for length, cnt in by_len.items():
        counts = list(cnt.values())
        lengths = huffman_lengths(counts)
        huff += sum(l * c for l, c in zip(lengths, counts))
        table += len(cnt) * (length + pos_bits(max(lengths) + 1)) + 16  # patterns + code lengths + table size
    unique = sum(len(c) for c in by_len.values())
    all_counts = Counter(s for s in symbols)
    repeated = sum(c for c in all_counts.values() if c > 1)
    top = ", ".join(f"{''.join(map(str, s)) or '-'}×{c}" for s, c in all_counts.most_common(6))
    print(f"  {title}: {len(symbols)} total, {unique} unique, {repeated / len(symbols):.0%} are repeats")
    print(f"    top: {top}")
    print(f"    inline {inline} bits | huffman per length {huff} + dictionary {table} = {huff + table} "
          f"({huff + table - inline:+d})")


def residual_stats(coder: GainCoder, x):
    width = coder.bit_count
    print("==B) leaf contents in the tree==")
    gain_leaves = [tuple(bit_of(int(t.value), p, width) for p in rem) for t, rem, _ in walk(coder) if kind(t) == LEAF]
    pattern_stats("GainCoder leaf residuals", gain_leaves)

    pw, _, stats = best_writer(coder, x)
    leaves, masks, pvals = [], [], []
    for t, rem, _, k in pw.walk():
        if k == LEAF:
            leaves.append(tuple(bit_of(int(t.value), p, width) for p in rem))
        elif k == PAIR:
            from gaintrie.pair_writer import leaf_values, pair_split
            a, b = leaf_values(t)
            diff, one = pair_split(a, b, rem, width)
            masks.append(tuple(int(i in diff) for i in range(len(rem))))
            pvals.append(tuple(bit_of(one, p, width) for i, p in enumerate(rem) if i != diff[0]))
    pattern_stats("PairWriter leaf residuals", leaves)
    pattern_stats("PairWriter pair masks (raw)", masks)
    pattern_stats("PairWriter pair values", pvals)


if __name__ == "__main__":
    for name, bits in (("model.layers.0.input_layernorm.weight.bin", 32),
                       ("model.layers.0.post_attention_layernorm.weight.bin", 32),
                       ("model.layers.0.input_layernorm.weight.bin", 16),
                       ("model.layers.0.post_attention_layernorm.weight.bin", 16)):
        x = np.frombuffer(sandbox_path("bins", name).read_bytes(), dtype=np.uint32 if bits == 32 else np.uint16)
        values, counts = np.unique(x, return_counts=True)
        coder = GainCoder(values, counts, bits)
        total = sum(gain_encode(coder, x)[1].values())
        print(f"######## {name} {bits}b (GainCoder total {total})")
        stream_stats(coder, x)
        residual_stats(coder, x)
        print("END")
