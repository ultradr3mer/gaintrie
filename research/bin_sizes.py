"""Bin size sweep: one stream per matrix kind (the kind's layer files chained
in layer order), cut into bins of a fixed size, every bin encoded on its own.
Reports the stream total in bits per value, so bin sizes compare directly.

Methods per bin, all written + decoded: GainCoder (tree_writer), PairWriter with
tree codes, PairWriter with the stream mode chosen per bin (tree codes or Huffman
over the leaves), FlipGainCoder flip+pair. Values are bf16, read as uint16.
"""
from pathlib import Path
import sys

import numpy as np

if globals().get("__package__", "") in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaintrie.GainCoder import GainCoder
from research.FlipGainCoder import FlipGainCoder, encode as flip_encode, decode as flip_decode
from research.leaf_stats import entropy_bits
from gaintrie.pair_writer import best_writer, decode as pair_decode, TREE_CODES, AUTO
from gaintrie.sandbox_paths import sandbox_path
from gaintrie.tree_writer import encode as gain_encode

BITS = 16
KINDS = ("input_layernorm", "post_attention_layernorm")
SIZES = (512, 1024, 2048, 4096, 8192, 16384, None)  # None = the whole stream as one bin
METHODS = ("GainCoder", "PairWriter", "flip+pair", "Pair+Huff")


def layer_of(path: Path) -> int:
    return int(path.name.split(".")[2])


def kind_stream(kind: str):
    """-> (all layers of the kind chained, layer count)"""
    paths = sorted(sandbox_path("bins").glob(f"model.layers.*.{kind}.weight.bin"), key=layer_of)
    return np.concatenate([np.fromfile(p, dtype=np.uint16) for p in paths]), len(paths)


def bin_costs(x) -> dict:
    values, counts = np.unique(x, return_counts=True)
    coder = GainCoder(values, counts, BITS)
    out = {"GainCoder": sum(gain_encode(coder, x)[1].values())}

    for label, stream in (("PairWriter", TREE_CODES), ("Pair+Huff", AUTO)):
        _, data, stats = best_writer(coder, x, stream)
        assert np.array_equal(pair_decode(data), x.astype(np.uint64)), "PairWriter round trip failed"
        out[label] = sum(stats.values())
    out["huffman"] = bool(stats["code_lengths"])

    data, fstats = flip_encode(FlipGainCoder(coder, "dp", 8), x)
    assert np.array_equal(flip_decode(data), x.astype(np.uint64)), "FlipGainCoder round trip failed"
    out["flip+pair"] = sum(fstats.values())

    out["unique"] = len(values)
    return out


def sweep(kind: str):
    x, layers = kind_stream(kind)
    n = len(x)
    _, counts = np.unique(x, return_counts=True)
    print(f"######## {kind}: {layers} layers, {n} values, {len(counts)} unique, "
          f"entropy of the whole stream {entropy_bits(counts.tolist()) / n:.3f} bits/value (raw {BITS})")
    print(f"{'bin size':>9}{'bins':>6}{'uniq/bin':>10}" + "".join(f"{m:>12}" for m in METHODS) + "   best")
    per_layer = None
    for size in SIZES:
        size = size or n
        bins = [x[i:i + size] for i in range(0, n, size)]
        costs = [bin_costs(b) for b in bins]
        totals = {m: sum(c[m] for c in costs) / n for m in METHODS}
        best = min(totals, key=totals.get)
        uniq = np.mean([c["unique"] for c in costs])
        huff = sum(c["huffman"] for c in costs)
        print(f"{size:>9}{len(bins):>6}{uniq:>10.0f}" + "".join(f"{totals[m]:>12.3f}" for m in METHODS)
              + f"   {best} (huffman in {huff}/{len(bins)} bins)")
        if size == 4096:
            per_layer = costs
    print("per layer (bin size 4096 = one layer file), bits/value:")
    for i, c in enumerate(per_layer):
        print(f"  layer {i}: unique {c['unique']:>5}  " + "  ".join(f"{m} {c[m] / 4096:.3f}" for m in METHODS))


if __name__ == "__main__":
    for kind in KINDS:
        sweep(kind)
    print("END")
