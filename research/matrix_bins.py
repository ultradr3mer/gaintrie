"""Bin size sweep on the big matrices (one layer), on sampled bins: a matrix has
far too many bins for pure Python, so per size `count` bins spread evenly over
the matrix are encoded with PairWriter (stream mode chosen per bin) and
compared with xz and the bin's order-0 entropy (no table). The first bin of
every size is decoded and checked. Values are bf16, read as uint16.
"""
from pathlib import Path
import lzma
import sys
import time

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.GainCoder import GainCoder
from gaintrie.pair_writer import best_writer, decode, AUTO
from gaintrie.sandbox_paths import sandbox_path

BITS = 16
LAYER = 0
MATRICES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.k_proj")
SIZES = {4096: 16, 16384: 8, 65536: 4, 262144: 2, 1048576: 1}  # bin size: sampled bins


def entropy_bits(x) -> float:
    c = np.bincount(x, minlength=1 << BITS)
    c = c[c > 0]
    return float((c * np.log2(len(x) / c)).sum())


def sample_bins(x, size, count):
    starts = np.linspace(0, len(x) // size - 1, count).astype(int) * size
    return [np.array(x[s:s + size]) for s in starts]


def sweep(matrix: str):
    x = np.memmap(sandbox_path("bins", f"model.layers.{LAYER}.{matrix}.weight.bin"), dtype=np.uint16, mode="r")
    n = len(x)
    print(f"######## {matrix} layer {LAYER}: {n:,} values, {np.count_nonzero(np.bincount(x, minlength=1 << BITS))} unique, "
          f"order-0 entropy of the whole matrix {entropy_bits(x) / n:.3f} bits/value (raw {BITS})")
    print(f"{'bin size':>9}{'sampled':>8}{'uniq/bin':>10}{'entropy':>9}{'xz':>8}{'ours':>8}{'huffman':>9}{'time':>7}")
    for size, count in SIZES.items():
        if size > n:
            continue
        t = time.time()
        total = ent = xz = uniq = huff = 0
        for i, b in enumerate(sample_bins(x, size, count)):
            values, counts = np.unique(b, return_counts=True)
            _, data, stats = best_writer(GainCoder(values, counts, BITS), b, AUTO)
            if i == 0:
                assert np.array_equal(decode(data), b.astype(np.uint64)), "round trip failed"
            total += sum(stats.values())
            ent += entropy_bits(b)
            xz += len(lzma.compress(b.tobytes(), preset=6)) * 8
            uniq += len(values)
            huff += bool(stats["code_lengths"])
        m = size * count
        print(f"{size:>9}{count:>8}{uniq / count:>10.0f}{ent / m:>9.3f}{xz / m:>8.3f}{total / m:>8.3f}"
              f"{huff:>6}/{count:<2}{time.time() - t:>6.0f}s", flush=True)


if __name__ == "__main__":
    for matrix in MATRICES:
        sweep(matrix)
    print("END")
