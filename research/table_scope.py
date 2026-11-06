"""Where do the bits go per bin size, and does a global Huffman table help small bins?

Per sampled bin (as matrix_bins), in bits per value:
  local     PairWriter + Huffman stream with the bin's own code length table
            (written; breakdown tree / length table / stream)
  tree      PairWriter with tree codes, no table beside the tree (written)
  hybrid    the bin's PairWriter tree defines which values occur; the stream codes
            are Huffman over the GLOBAL counts of exactly those values, so the bin
            carries no length table (computed; the global count table is sent once)
  global    no per-bin tree: one dictionary + Huffman table for the whole matrix
            (computed: global PairWriter tree + length table, amortized over the matrix)
"""
from pathlib import Path
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.GainCoder import GainCoder
from gaintrie.code_utils import huffman_lengths
from research.matrix_bins import sample_bins, entropy_bits, BITS, LAYER
from gaintrie.pair_writer import best_writer, code_length_bits, TREE_CODES, HUFFMAN
from gaintrie.sandbox_paths import sandbox_path

MATRICES = ("mlp.gate_proj", "self_attn.k_proj")
SIZES = {4096: 16, 16384: 8, 65536: 4, 262144: 2, 1048576: 1}


def tree_bits(stats) -> int:
    return sum(v for k, v in stats.items() if k not in ("stream", "code_lengths", "header"))


def sweep(matrix: str):
    x = np.memmap(sandbox_path("bins", f"model.layers.{LAYER}.{matrix}.weight.bin"), dtype=np.uint16, mode="r")
    n = len(x)
    gcount = np.bincount(x, minlength=1 << BITS)
    gvalues = np.flatnonzero(gcount)
    glen = np.zeros(1 << BITS, dtype=np.int64)
    glen[gvalues] = huffman_lengths(gcount[gvalues].tolist())
    # global dictionary: PairWriter tree over all values + one length table (+ the counts for the hybrid)
    _, _, gstats = best_writer(GainCoder(gvalues, gcount[gvalues], BITS), gvalues, TREE_CODES)
    gdict = tree_bits(gstats) + code_length_bits(glen[gvalues].tolist())
    global_bits = (float((glen * gcount).sum()) + gdict) / n
    print(f"######## {matrix} layer {LAYER}: {n:,} values, {len(gvalues)} unique, "
          f"entropy {entropy_bits(x) / n:.3f}, global Huffman {global_bits:.3f} bits/value "
          f"(dictionary+table {gdict} bits = {gdict / n:.4f}/value)")
    print(f"{'bin size':>9}{'uniq':>6} | {'local: tree':>12}{'table':>7}{'stream':>8}{'total':>8}"
          f" | {'tree codes':>10} | {'hybrid: tree':>13}{'stream':>8}{'total':>8} | {'global':>7}")
    for size, count in SIZES.items():
        acc = np.zeros(8)
        for b in sample_bins(x, size, count):
            values, counts = np.unique(b, return_counts=True)
            coder = GainCoder(values, counts, BITS)
            _, _, ls = best_writer(coder, b, HUFFMAN)
            _, _, ts = best_writer(coder, b, TREE_CODES)
            hyb_len = huffman_lengths(gcount[values].tolist())
            acc += [len(values), tree_bits(ls) + ls["header"], ls["code_lengths"], ls["stream"],
                    sum(ts.values()), tree_bits(ts) + ts["header"],
                    sum(l * c for l, c in zip(hyb_len, counts.tolist())), 0]
        m = size * count
        u, lt, lc, lsd, tc, ht, hs, _ = acc
        print(f"{size:>9}{u / count:>6.0f} | {lt / m:>12.3f}{lc / m:>7.3f}{lsd / m:>8.3f}{(lt + lc + lsd) / m:>8.3f}"
              f" | {tc / m:>10.3f} | {ht / m:>13.3f}{hs / m:>8.3f}{(ht + hs) / m + gdict / n:>8.3f} | {global_bits:>7.3f}",
              flush=True)


if __name__ == "__main__":
    for matrix in MATRICES:
        sweep(matrix)
    print("END")
